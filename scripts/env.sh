#!/bin/bash
# ============================================================
# Environment variables for Instant Policy
#
# Usage: source scripts/env.sh
# ============================================================

# CoppeliaSim (required for RLBench / PerAct2 data collection)
export COPPELIASIM_ROOT=$HOME/langtian/tools/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# ShapeNet data path
export SHAPENET_ROOT=$HOME/zhewen/instant_policy/data/shapenet

# Conda environment
export CONDA_ENV_NAME=ip_env

# CUDA (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

# Virtual display for headless rendering (uncomment for headless servers)
# Xvfb :99 -screen 0 1280x1024x24 &
# export DISPLAY=:99
