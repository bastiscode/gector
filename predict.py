import argparse
import time
from typing import List

import spacy
from spacy.lang.en import TOKENIZER_EXCEPTIONS

from tqdm import tqdm

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel

nlp = spacy.load("en_core_web_sm")


def fix_sequence(sequence: str) -> str:
    whitespaces = []
    doc = nlp(sequence)
    words = [token.text for token in doc]
    num_quotes = 0
    for i, token in enumerate(doc):
        next_token = doc[i + 1] if i < len(doc) - 1 else None

        add_with_ws = True

        if next_token is not None:
            if token.is_quote:
                num_quotes += 1
                if num_quotes % 2 == 1:
                    add_with_ws = False

            if next_token.is_quote and num_quotes % 2 == 1:
                add_with_ws = False

            if token.text + next_token.text in TOKENIZER_EXCEPTIONS:
                add_with_ws = False

            if (token.is_left_punct and not token.is_quote) or token.text == "-":
                add_with_ws = False

            if (next_token.is_right_punct and not next_token.is_quote) \
                    or (next_token.is_punct and not next_token.is_right_punct and not next_token.is_left_punct):
                add_with_ws = False

            if next_token.text == "'s":
                add_with_ws = False

        else:
            add_with_ws = False

        whitespaces.append(add_with_ws)

    out_sequence = ""
    for word, ws in zip(words, whitespaces):
        out_sequence += word + ws * " "
    return out_sequence


def predict_for_file(input_file, output_file, model, batch_size=32, to_normalize=False):
    test_data = read_lines(input_file)
    indices = list(range(len(test_data)))
    test_data, input_indices = list(zip(*sorted(list(zip(test_data, indices)), key=lambda e: len(e[0]), reverse=True)))

    input_test_data = []
    for line, idx in zip(test_data, indices):
        doc = nlp(line)
        input_test_data.append(" ".join([token.text for token in doc]))

    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in tqdm(input_test_data, desc="predicting file"):
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    # reorder
    result_lines_out: List[str] = ["" for _ in range(len(test_data))]
    for idx, result_line in zip(input_indices, result_lines):
        result_line = fix_sequence(result_line)
        result_lines_out[idx] = result_line

    with open(output_file, 'w') as f:
        f.write("\n".join(result_lines_out) + "\n")
    return cnt_corrections


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    start = time.perf_counter()
    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size, 
                                       to_normalize=args.normalize)
    end = time.perf_counter()
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}, runtime: {end - start:.2f}s")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()
    main(args)
