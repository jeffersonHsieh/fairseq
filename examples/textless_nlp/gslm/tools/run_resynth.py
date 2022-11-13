# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import logging
import os
from pathlib import Path
import progressbar
from time import time
import joblib
import soundfile as sf
import torch
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import get_feature_reader
from examples.textless_nlp.gslm.unit2speech.tts_data import TacotronInputDataset
from examples.textless_nlp.gslm.unit2speech.utils import (
    load_tacotron,
    load_waveglow,
    synthesize_audio,
)


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(((str(Path(f).stem)), os.path.join(root, f)))
    return out

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description="GSLM U2S tool")
    parser.add_argument('input_dir', type=str,
                    help='Path to the input dir.')
    parser.add_argument('output_dir', type=str,
                    help='Path to the output dir.')
    parser.add_argument('--extension', type=str, default='wav',
                    help='Audio file extension in the input. Default: wav')

    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        required=False,
        help="run on subset of x samples",
    )
    parser.add_argument(
      "--subset_file",
      type=str,
      default=None,
      required=False,
      help="path to manifest file with files to use"
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint",
    )
    parser.add_argument("--layer", type=int, help="Layer of acoustic model")
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--tts_model_path",
        type=str,
        help="TTS model file path to use for inference",
    )
    parser.add_argument(
        "--code_dict_path",
        type=str,
        help="Code dict file path to use for inference",
    )
    parser.add_argument(
        "--waveglow_path",
        type=str,
        help="Waveglow (vocoder) model file path to use for inference",
    )
    parser.add_argument("--max_decoder_steps", type=int, default=2000)
    parser.add_argument("--denoiser_strength", type=float, default=0.1)
    return parser


################################################
def main(args, logger):
    # Acoustic Model
    logger.info(f"Loading acoustic model from {args.tts_model_path}...")
    feature_reader_cls = get_feature_reader(args.feature_type)
    reader = feature_reader_cls(
        checkpoint_path=args.acoustic_model_path, layer=args.layer
    )
    # K-means Model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    # TTS Model
    logger.info(f"Loading TTS model from {args.tts_model_path}...")
    tacotron_model, sample_rate, hparams = load_tacotron(
        tacotron_model_path=args.tts_model_path,
        max_decoder_steps=args.max_decoder_steps,
    )

    # Waveglow Model
    logger.info(f"Loading Waveglow model from {args.waveglow_path}...")
    waveglow, denoiser = load_waveglow(waveglow_path=args.waveglow_path)



    # Dataset
    if not os.path.exists(hparams.code_dict):
        hparams.code_dict = args.code_dict_path
    tts_dataset = TacotronInputDataset(hparams)
    print(f"Finding all audio files with extension '{args.extension}' from {args.input_dir}...")
    audio_files = find_all_files(args.input_dir, args.extension)
    print(f"Done! Found {len(audio_files)} files.")

    if args.subset:
      audio_files = audio_files[:args.subset]
    
    if args.subset_file:
      subset = set()
      with open(args.subset_file) as f:
        for line in f:
          fname,_ = line.rstrip().split("\t")
          subset.add(fname)
      audio_files = [fname for fname in audio_files if os.path.basename(fname[1]) in subset]


    # Convert to relative path
    audio_files = [os.path.relpath(file[-1], start=args.input_dir) for file in audio_files]

    # Create all the directories needed
    rel_dirs_set = set([os.path.dirname(file) for file in audio_files])
    for rel_dir in rel_dirs_set:
        Path(os.path.join(args.output_dir, rel_dir)).mkdir(parents=True, exist_ok=True)

    iters = 0
    bar = progressbar.ProgressBar(maxval=len(audio_files))
    bar.start()
    start_time = time()
    for index, file in enumerate(audio_files):
        bar.update(index)
        in_file_path = os.path.join(args.input_dir, file)
        out_file_path = os.path.join(args.output_dir, os.path.splitext(file)[0]+".wav")
        feats = reader.get_feats(in_file_path).cpu().numpy()
        iters += 1
        if iters%50==0:
            gc.collect()
            torch.cuda.empty_cache()

        quantized_units = kmeans_model.predict(feats)
        quantized_units_str = " ".join(map(str, quantized_units))

        tts_input = tts_dataset.get_tensor(quantized_units_str)
        mel, aud, aud_dn, has_eos = synthesize_audio(
            tacotron_model,
            waveglow,
            denoiser,
            tts_input.unsqueeze(0),
            strength=args.denoiser_strength,
        )
        sf.write(f"{out_file_path}", aud_dn[0].cpu().float().numpy(), sample_rate)
        # logger.info("Resynthesis done!\n")
    bar.finish()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
