
# Overview

Data preprocessing is often glossed over in MT papers, 
but seemingly small changes can have large impacts on downstream MT performance.


These scripts perform the bitext filtering used to train the [Prism](https://github.com/thompsonb/prism) model, in particular:
 - [LASER margin filtering](https://arxiv.org/abs/1811.01136) [if both languages are supported by [LASER](https://github.com/facebookresearch/LASER)]
 - N-gram overlap filtering
 - Sentence-level [language ID](https://fasttext.cc/docs/en/language-identification.html) filtering
 - Average n-gram [language ID](https://fasttext.cc/docs/en/language-identification.html) filtering
 - Sentence length filtering

We release these scripts in the hope that they may be helpful for other researchers. 

Sentence length and n-grams are defined in terms of subword units.
This was done so that the methods would generalize to languages that do not denote word boundaries with whitespace. 
This repo includes the sentencepiece model which we used for filtering, which should work for most of the LASER languages. You may wish to replace this with your own model.

Note that while LASER embeddings can be computed on a CPU, it is quite slow. We recommend running on a GPU for any reasonably sized dataset.

Note that we also release the [filtered data](http://data.statmt.org/prism/prism_data.tz) used to train the Prism model. 

# Installation

Create conda environment, install dependencies, and download LASER and LID model: (note:[you will need gcc](https://github.com/facebookresearch/fastText/issues/1196) to run these commands)
```bash
conda create -y --name prismfilt python=3.8 faiss-gpu==1.7.1 scipy==1.6.2 pytorch==1.9.0 sentencepiece==0.1.95 pandas==1.2.4 -c pytorch -c anaconda -c conda-forge # faiss-cpu on CPU
conda activate prismfilt  # older conda versions: `source activate prismfilt`
pip install fasttext==0.9.2
pip install laserembeddings==1.1.2
pip install laserembeddings[zh,ja]==1.1.2
python -m laserembeddings download-models
wget -O lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
# Scoring

Compute scores for input sentence pairs: (this will use a GPU if available)
```bash
python score.py \
   --src_file test.en \
   --tgt_file test.de \
   --src_lang en \
   --tgt_lang de \
   --out_file test.pkl
```
The sentence pairs and their corresponding scores are written to a pickle file to be used below. 
Note there is no distinction between src and tgt languages, except to keep track of which is which. 

To see an example of the scores, run the following in a python prompt:
```python
import pandas
from pprint import pprint
pprint(pandas.read_pickle('test.pkl').iloc[0].to_dict())
```

Which should print something like this:
```
{'laser_score': 1.33841,
 'overlap_frac_3gram': 0.0,
 'overlap_frac_4gram': 0.0,
 's_len': 25,
 's_lid_chunk_score': 1.0,
 's_lid_score': 0.9667384,
 'src': 'For instance, the central banks of the US, Europe, Japan, and Britain '
        'could accept Brazilian paper at their discount windows.',
 't_len': 26,
 't_lid_chunk_score': 0.8636364,
 't_lid_score': 0.9862412,
 'tgt': 'Zum Beispiel könnten die Zentralbanken der USA, Europas, Japans und '
        'Großbritanniens brasilianische Wertpapiere an ihren Diskontschaltern '
        'akzeptieren.'}
```

Note that LASER margin scoring is only applied if both languages are supported by LASER.
Also note that some basic filtering (LID, n-gram overlap, and length) are applied prior to 
the LASER margin scoring, as we find that 
duplicates and sentences in the same language can cause issues with margin scoring, 
and extremely long sentences can hang the LASER embedding code. For more information, run `score.py -h`



# Filtering

To filter the sentence pairs using on the previously computed scores, run:
```bash
python filter.py --score_file test.pkl --src_clean clean.en --tgt_clean clean.de
```

Note that the output files will be deduplicated and shuffled. 

All thresholds have defaults, which can be overwritten. To see the flags and defaults, run:
```bash
python filter.py -h
```
Which should produce:
```
usage: filter.py [-h] --score_file SCORE_FILE --src_clean SRC_CLEAN --tgt_clean TGT_CLEAN [--min_len MIN_LEN] [--max_len MAX_LEN] [--max_3gram_overlap MAX_3GRAM_OVERLAP]
                 [--max_4gram_overlap MAX_4GRAM_OVERLAP] [--min_laser_score MIN_LASER_SCORE] [--min_lid_score MIN_LID_SCORE] [--min_chunk_lid_score MIN_CHUNK_LID_SCORE]

optional arguments:
  -h, --help            show this help message and exit
  --score_file SCORE_FILE
                        Input file from score.py (default: None)
  --src_clean SRC_CLEAN
                        Output clean sarget file name (default: None)
  --tgt_clean TGT_CLEAN
                        Output clean target file name (default: None)
  --min_len MIN_LEN     Minimum allowable sentence length (default: 1)
  --max_len MAX_LEN     Maximum allowable sentence length (default: 200)
  --max_3gram_overlap MAX_3GRAM_OVERLAP
                        Maximum allowable fraction of 3-gram overlap (default: 0.6)
  --max_4gram_overlap MAX_4GRAM_OVERLAP
                        Maximum allowable fraction of 4-gram overlap (default: 0.4)
  --min_laser_score MIN_LASER_SCORE
                        Minimum allowable LASER margin score (default: 1.04)
  --min_lid_score MIN_LID_SCORE
                        Minimum allowable sentence-level language ID score (default: 0.5)
  --min_chunk_lid_score MIN_CHUNK_LID_SCORE
                        Minimum allowable average of 5-gram language ID scores (default: 0.5)
```

If your data has more than a few million lines, you will likely want to break up your data into chunks and process them independently. This is due to the LASER margin filtering performing nearest neighbor search, which compares every source vector to every target vector. Anecdotally, processing a few million lines at a time seems to produce about the same results as processing an entire file at once, without running into computational issues. 


# Publications

If you use these these scripts in your work, please cite our [EMNLP paper](https://www.aclweb.org/anthology/2020.emnlp-main.8/):
```
@inproceedings{thompson-post-2020-automatic,
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
