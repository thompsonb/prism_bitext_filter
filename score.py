#!/usr/bin/env python

import argparse
import hashlib
import html
import multiprocessing 
import os
import shutil
import urllib.request
from collections import Counter

import faiss
import fasttext
import pandas
import numpy as np
import sentencepiece as spm
import torch

from laserembeddings import Laser
from scipy.spatial.distance import cosine as cosine_dist


# modified from https://github.com/facebookresearch/LASER/blob/main/source/mine_bitexts.py
def knnGPU(x, y, k, mem=5*1024*1024*1024):
    dim = x.shape[1]
    batch_size = mem // (dim*4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            idx = faiss.IndexFlatIP(dim)
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
                idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind


laser_langcodes = ['af', 'sq', 'am', 'ar', 'hy', 'ay', 'az', 'eu', 'be', 'bn', 'ber', 'bs', 'br', 
                   'bg', 'my', 'ca', 'km', 'cbk', 'zh', 'kzj', 'kw', 'hr', 'cs', 'da', 'nl', 'mhr', 
                   'en', 'eo', 'et', 'fi', 'fr', 'gl', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 
                   'is', 'io', 'id', 'ia', 'ie', 'ga', 'it', 'ja', 'kab', 'kk', 'ko', 'ku', 'lv', 
                   'la', 'lfn', 'lt', 'nds', 'mk', 'mg', 'ms', 'ml', 'dv', 'mr', 'nb', 'oc', 'fa', 
                   'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'sw', 'sv', 
                   'tl', 'tg', 'ta', 'tt', 'te', 'th', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi', 'wuu', 'yue']

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
lid_model_file = os.path.join(__location__, f'lid.176.bin')

laser = Laser()

lid_model = fasttext.load_model(lid_model_file)

margin_score_num_neighbors = 5

faiss.omp_set_num_threads(multiprocessing.cpu_count())

sp_model = os.path.join(__location__, 'spmTemp200k.model')  # model used just for preprocessing

sppp = spm.SentencePieceProcessor()
sppp.Load(sp_model)


def tokenize_with_spm(s):
    return sppp.EncodeAsPieces(s)


def detok(lst):
    return ''.join(lst).replace('▁', ' ')


def check_lang(tokenized_line, lang, ngram_size=4, num_langs=10):
    v, p = lid_model.predict(detok(tokenized_line), k=num_langs)

    lang = lang.lower()  # make sure lower case lang code

    try:
        sent_score = p[v.index(f'__label__{lang}')]
    except ValueError:
        sent_score = 0.0

    lang_scores = []
    for ngram in ngrams(tokenized_line, num_ngrams=ngram_size):
        chunk = detok(ngram)

        v, p = lid_model.predict(chunk, k=1)

        if v[0] == f'__label__{lang}':
            chunk_lang_score = 1
        else:
            chunk_lang_score = 0
        lang_scores.append(chunk_lang_score)

    if len(lang_scores) > 0:
        chunk_score = np.mean(lang_scores)
    else:
        chunk_score = 0.0

    return sent_score, chunk_score


def clean_line(line):
    # global voices has html garbage in it like &middot; &#8212 &amp;  etc
    # it also has '&middot; Global Voices' over and over and over
    line = line.strip().replace('&middot; Global Voices', '')
    line = html.unescape(line)

    # Apparently there are some html-escaped end-of-line characters in the data...
    line = line.replace('\n', ' ')
    line = line.replace('\b', ' ')
    line = line.replace('\t', ' ')
    line = line.replace('\r', ' ')

    line = line.strip()

    return line


def overlap_with_dups(lst0, lst1):
    # Counter can be used to give intersection which handles duplicates
    shared_words_with_dups = list((Counter(lst0) & Counter(lst1)).elements())
    frac_overlap = len(shared_words_with_dups) / max(1, min(len(lst0), len(lst1)))
    return frac_overlap


def ngrams(list_of_tokens, num_ngrams):
    grams = []
    for ii in range(len(list_of_tokens) - num_ngrams + 1):
        gram = list_of_tokens[ii:ii + num_ngrams]
        grams.append(tuple(gram))
    return grams


def spm_ngram_overlap_frac(sent0, sent1, num_ngrams=3):
    """
    about 1M lines per min
    """
    ngrams0 = ngrams(list_of_tokens=sent0, num_ngrams=num_ngrams)
    ngrams1 = ngrams(list_of_tokens=sent1, num_ngrams=num_ngrams)
    return overlap_with_dups(ngrams0, ngrams1)


def compute_hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()


def normalize(emb):
    for ii in range(emb.shape[0]):
        emb[ii, :] = emb[ii, :] / (np.linalg.norm(emb[ii, :]) + 1e-5)
    return emb


def make_blob(stuff,):
    src_line, tgt_line, src_lang, tgt_lang = stuff
    
    blob = dict()

    # strip out some html/globalvoices garbage
    s_clean = clean_line(src_line)
    t_clean = clean_line(tgt_line)
    blob['src'] = s_clean
    blob['tgt'] = t_clean

    # tokenize (sort of) with sentencepiece
    # this is only used for length and n-gram overlap
    # this is intended to handle languages without whitespace word boundaries
    s_tok = tokenize_with_spm(src_line)
    t_tok = tokenize_with_spm(tgt_line)
    
    # git LID scores (both at sentence level and chunk levels)
    s_lid_score, s_lid_chunk_score = check_lang(s_tok, src_lang, ngram_size=5)
    blob['s_lid_score'] = s_lid_score
    blob['s_lid_chunk_score'] = s_lid_chunk_score
    
    t_lid_score, t_lid_chunk_score = check_lang(t_tok, tgt_lang, ngram_size=5)
    blob['t_lid_score'] = t_lid_score
    blob['t_lid_chunk_score'] = t_lid_chunk_score
    
    # get n-gram overlap between source and target
    for gram in (3, 4):
        blob[f'overlap_frac_{gram}gram'] = spm_ngram_overlap_frac(s_tok, t_tok, num_ngrams=gram)

    # save off length in (sentencepiece) tokens
    blob['s_len'] = len(s_tok)
    blob['t_len'] = len(t_tok)
    
    blob['s_hash'] = compute_hash(s_clean)
    blob['t_hash'] = compute_hash(t_clean)
    
    return blob



def score(src_lines, tgt_lines, src_lang, tgt_lang, do_laser, LID_threshold, four_gram_threshold, max_LASER_sent_len):
    blobs = []

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        blobs = pool.map(make_blob, [(src_line, tgt_line, src_lang, tgt_lang) for src_line, tgt_line in zip(src_lines, tgt_lines)])

    
    # we do not want duplicates in our embeddings, because it will throw off margin scoring
    s_to_embed = set()
    t_to_embed = set()
    
    for blob in blobs:        
        # save off sentences to embed with laser
        # filter out extreamly long sentences (probably javascrpt or something) so LASER does not hang 
        # also filter out blank lines
        # also filter out lines in the wrong language, which can confuse the margin scores
        # also filter out near copies, which can confuse the margin scores
        if blob['overlap_frac_4gram'] < four_gram_threshold:
            if 0 < blob['s_len'] < max_LASER_sent_len and blob['s_lid_score'] > LID_threshold:
                s_to_embed.add(blob['src'])

            if 0 < blob['t_len'] < max_LASER_sent_len and blob['t_lid_score'] > LID_threshold:
                t_to_embed.add(blob['tgt'])


    if do_laser and len(s_to_embed) and len(t_to_embed):
        print('using GPU?', torch.cuda.is_available(), flush=True)            
        print('Computing LASER embeddings', flush=True)
        # LASER embeddings

        s_to_embed = list(s_to_embed)
        s_embed = normalize(laser.embed_sentences(s_to_embed, lang=src_lang))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            hashes = pool.map(compute_hash, s_to_embed)

        s_hash2idx = {h: ii for ii, h in enumerate(hashes)}

        t_to_embed = list(t_to_embed)
        t_embed = normalize(laser.embed_sentences(t_to_embed, lang=tgt_lang))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            hashes = pool.map(compute_hash, t_to_embed)

        t_hash2idx = {h: ii for ii, h in enumerate(hashes)}

        print('Performing nearest neighbor search', flush=True)

        t_sims, t_idxs = knnGPU(t_embed, s_embed, margin_score_num_neighbors, mem=5*1024*1024*1024)
        s_sims, s_idxs = knnGPU(s_embed, t_embed, margin_score_num_neighbors, mem=5*1024*1024*1024)
        
        s_neighbor_sim = (2.0 - s_sims) / 2.0
        t_neighbor_sim = (2.0 - t_sims) / 2.0

        # make sure FAISS actually found neighbors... 
        t_fails = np.any(t_idxs==-1, axis=1) # faiss pads with -1 when it does not find anything
        if sum(t_fails):
            print(f'WARNING!!! Failed to find {margin_score_num_neighbors} tgt neighbors for {sum(t_fails)} of {len(t_fails)} vectors', flush=True)

        s_fails = np.any(s_idxs==-1, axis=1) # faiss pads with -1 when it does not find anything
        if sum(s_fails):
            print(f'WARNING!!! Failed to find {margin_score_num_neighbors} src neighbors for {sum(s_fails)} of {len(s_fails)} vectors', flush=True)
                    
        # Average the neighbor similarity. /2 so can add in scoring function
        s_norm = s_neighbor_sim.mean(axis=1) / 2.0
        t_norm = t_neighbor_sim.mean(axis=1) / 2.0

        # LASER margin scoring
        for ii, blob in enumerate(blobs):
            try:
                s_idx = s_hash2idx[blob['s_hash']]
                t_idx = t_hash2idx[blob['t_hash']]
                cos_sim = 1.0 - cosine_dist(s_embed[s_idx], t_embed[t_idx])
                laser_score = cos_sim / (s_norm[s_idx] + t_norm[t_idx])
            except KeyError:
                laser_score = 0.0
            blob['laser_score'] = laser_score

    return blobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--src_file', type=str, required=True, help='Input source file')
    parser.add_argument('--tgt_file', type=str, required=True, help='Input target file')
    parser.add_argument('--src_lang', type=str, required=True, help='Input source language code')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Input target language code')
    parser.add_argument('--out_file', type=str, required=True, help='Output file (pickled pandas DataFrame)')
    parser.add_argument('--LID_threshold', type=float, default=0.5, help='LASER will only run on sentences with LID scores greater than this value. Sentence pairs in the same language can confuse LASER margin scoring. (default: %(default)s)')
    parser.add_argument('--max_LASER_sent_len', type=int, default=2000, help='LASER will only run sentence with length (in subwords) less than this value. Extreamly long sentences can cause LASER to hang. (default: %(default)s)')
    parser.add_argument('--four_gram_threshold', type=float, default=0.6, help='LASER will only run sentence pairs with 4-gram overlap (at the subword leve) below this value. (Near)duplicate sentences can confuse LASER margin scoring. (default: %(default)s)')

    args = parser.parse_args()

    do_laser = args.src_lang in laser_langcodes and args.tgt_lang in laser_langcodes
    print('Both langs supported by LASER:', do_laser, flush=True)

    
    blobs = score(src_lines=open(args.src_file, 'rt').readlines(),
                  tgt_lines=open(args.tgt_file, 'rt').readlines(),
                  src_lang=args.src_lang,
                  tgt_lang=args.tgt_lang,
                  do_laser=do_laser,
                  LID_threshold=args.LID_threshold,
                  four_gram_threshold=args.four_gram_threshold,
                  max_LASER_sent_len=args.max_LASER_sent_len)

    df = pandas.DataFrame(blobs)

    for name in ('s_hash', 't_hash'):
        df.drop(name, axis=1, inplace=True)

    for name in ('src', 'tgt'):
        df[name] = df[name].astype('string')

    for name in ('s_lid_score', 's_lid_chunk_score', 't_lid_score', 't_lid_chunk_score', 'overlap_frac_3gram', 'overlap_frac_4gram'):
        df[name] = df[name].astype('float32')

    if 'laser_score' in df:
        df['laser_score'] = df['laser_score'].astype('float32')

    df.to_pickle(args.out_file)
