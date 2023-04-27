#!/bin/bash

set -e
set -x

CEPSTRUM_MODELS=("MLP" "WaveRNN" "WaveLSTM" "SimpleLSTM" "ShallowCNN" "TSSD")
WAVE_MODELS=("WaveRNN" "WaveLSTM" "TSSD")

CEPSTRUM_FEATURE_CLASS=("lfcc" "mfcc")
WAVE_FEATURE_CLASS=("wave")

CEPSTRUM_TRAINED_MODELS=("MLP_lfcc_I" "MLP_mfcc_I" "SimpleLSTM_lfcc_I" "SimpleLSTM_mfcc_I" "ShallowCNN_lfcc_I" "ShallowCNN_mfcc_I")
WAVE_TRAINED_MODELS=("WaveRNN_wave_I" "WaveLSTM_wave_I" "TSSD_wave_I")

TESTS=("for2sec" "for-rerecorded" "for-2sec-modif/mp3" "for-2sec-modif/ogg" "for-2sec-modif/wma" "for-2sec-modif/m4v" "for-2sec-modif/reduce" "for-2sec-modif/volume" "for-2sec-modif/bitrate" "for-2sec-modif/white" "for-2sec-modif/street" "for-2sec-modif/birds" "for-2sec-modif/downsample" "for-2sec-modif/flanger" "for-2sec-modif/overdrive" "for-2sec-modif/reverb")

EPOCHS=20
BATCH_SIZE=256
SEED=42
CUDA_VISIBLE_DEVICES=0

TRAINING_DIR="for2sec/training"
VALIDATING_DIR="for2sec/validate"

TRAINING_COMMAND="python3.8 train.py --batch_size $BATCH_SIZE --epochs $EPOCHS --seed $SEED --deterministic"
TESTING_COMMAND="python3.8 train.py --eval_only --restore --seed $SEED --deterministic"

# TRAIN CEPSTRUM MODELS
for model in "${CEPSTRUM_MODELS[@]}";
do
    for feature_class in "${CEPSTRUM_FEATURE_CLASS[@]}";
    do
        $TRAINING_COMMAND --training $TRAINING_DIR --validation $VALIDATING_DIR --feature_classname $feature_class --model_classname $model
    done
done

# EVALUATE CEPSTRUM DATASETS
for model in "${CEPSTRUM_TRAINED_MODELS[@]}";
do
    for test in "${TESTS[@]}";
    do
        model_classname=$(echo "$test" | awk -F "_" '{print $1}')
        feature_class=$(echo "$test" | awk -F "_" '{print $2}')

        $TESTING_COMMAND --testing $test --feature_classname $feature_class --model_classname $model_classname
    done
done

# TRAIN WAVE MODELS
for model in "${WAVE_MODELS[@]}";
do
    for feature_class in "${WAVE_FEATURE_CLASS[@]}";
    do
        $TRAINING_COMMAND --training $TRAINING_DIR --validation $VALIDATING_DIR --feature_classname $feature_class --model_classname $model
    done
done

# EVALUATE WAVE DATASETS
for model in "${WAVE_TRAINED_MODELS[@]}";
do
    for test in "${TESTS[@]}";
    do
        model_classname=$(echo "$test" | awk -F "_" '{print $1}')
        feature_class=$(echo "$test" | awk -F "_" '{print $2}')

        $TESTING_COMMAND --testing $test --feature_classname $feature_class --model_classname $model_classname
    done
done