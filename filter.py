#!/usr/bin/env python

import argparse
import pandas 
from random import shuffle, seed

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    parser.add_argument('--score_file', type=str, required=True, help='Input file from score.py')
    parser.add_argument('--src_clean', type=str, required=True, help='Output clean sarget file name')
    parser.add_argument('--tgt_clean', type=str, required=True, help='Output clean target file name')

    # filter thresholds
    parser.add_argument('--min_len', type=int, default=1, help='Minimum allowable sentence length (default: %(default)s)' )
    parser.add_argument('--max_len', type=int, default=200, help='Maximum allowable sentence length (default: %(default)s)' )
    parser.add_argument('--max_3gram_overlap', type=float, default=0.6, help='Maximum allowable fraction of 3-gram overlap (default: %(default)s)' )
    parser.add_argument('--max_4gram_overlap', type=float, default=0.4, help='Maximum allowable fraction of 4-gram overlap (default: %(default)s)' )
    parser.add_argument('--min_laser_score', type=float, default=1.04, help='Minimum allowable LASER margin score (default: %(default)s)' )
    parser.add_argument('--min_lid_score', type=float, default=0.5, help='Minimum allowable sentence-level language ID score (default: %(default)s)' )
    parser.add_argument('--min_chunk_lid_score', type=float, default=0.5, help='Minimum allowable average of 5-gram language ID scores (default: %(default)s)' )

    args = parser.parse_args()

    df = pandas.read_pickle(args.score_file)

    filt = f't_len >= {args.min_len} & t_len <= {args.max_len} & s_len >= {args.min_len} & s_len <= {args.max_len} '
    filt += f' & overlap_frac_3gram <= {args.max_3gram_overlap} & overlap_frac_4gram <= {args.max_4gram_overlap} '
    filt += f' &  t_lid_chunk_score >= {args.min_chunk_lid_score} & s_lid_chunk_score >= {args.min_chunk_lid_score} '
    filt += f' &  t_lid_score >= {args.min_lid_score} & s_lid_score >= {args.min_lid_score}  '

    if 'laser_score' in df:
        filt += f' & laser_score >= {args.min_laser_score} '

    df2 = df.query(filt)

    pairs = list(zip(df2['src'].values, df2['tgt'].values))
        
    # de-duplicate, preserve order (dict insertion is ordered in 3.6+)
    pairs = list(dict.fromkeys(pairs))

    seed(0)
    shuffle(pairs)

    with open(args.src_clean, 'wt') as s_out, open(args.tgt_clean, 'wt') as t_out:
        for s, t in pairs:
            s_out.write(s + '\n')
            t_out.write(t + '\n')

    print(f'Filtered from {len(df):,} to {len(pairs):,} sentence pairs')
