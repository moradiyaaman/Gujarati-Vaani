#!/bin/bash
#
# Concurrent Curriculum Training Launcher
# Runs both male and female voice training simultaneously
#

echo "=========================================="
echo "🎓 CONCURRENT CURRICULUM TRAINING"
echo "   Male + Female voices in parallel"
echo "=========================================="
echo ""

# Configuration
EPOCHS_PER_STAGE=2
TOTAL_STAGES=4
SAMPLE_EVERY=200
LEARNING_RATE=1e-5

cd /home/azureuser/cloudfiles/code/Gujarati-Vaani

# Create log directory
mkdir -p training/logs

# Start Male voice training in background
echo "🚀 Starting MALE voice training..."
nohup python training/train_curriculum.py \
    --voice male \
    --epochs-per-stage $EPOCHS_PER_STAGE \
    --total-stages $TOTAL_STAGES \
    --sample-every $SAMPLE_EVERY \
    --lr $LEARNING_RATE \
    > training/logs/male_curriculum.log 2>&1 &

MALE_PID=$!
echo "   PID: $MALE_PID"
echo "   Log: training/logs/male_curriculum.log"
echo ""

# Small delay to prevent resource conflicts at startup
sleep 5

# Start Female voice training in background
echo "🚀 Starting FEMALE voice training..."
nohup python training/train_curriculum.py \
    --voice female \
    --epochs-per-stage $EPOCHS_PER_STAGE \
    --total-stages $TOTAL_STAGES \
    --sample-every $SAMPLE_EVERY \
    --lr $LEARNING_RATE \
    > training/logs/female_curriculum.log 2>&1 &

FEMALE_PID=$!
echo "   PID: $FEMALE_PID"
echo "   Log: training/logs/female_curriculum.log"
echo ""

echo "=========================================="
echo "✅ Both trainings started!"
echo ""
echo "📊 Training Configuration:"
echo "   • Epochs per stage: $EPOCHS_PER_STAGE"
echo "   • Total stages: $TOTAL_STAGES (Easy→Medium→Hard→Expert)"
echo "   • Total epochs: $((EPOCHS_PER_STAGE * TOTAL_STAGES)) per voice"
echo "   • Sample every: $SAMPLE_EVERY batches"
echo ""
echo "📁 Output Directories:"
echo "   • Male models:   training/curriculum_models/male/"
echo "   • Female models: training/curriculum_models/female/"
echo "   • Male samples:  training/samples/male_curriculum/"
echo "   • Female samples: training/samples/female_curriculum/"
echo ""
echo "📋 Monitor Commands:"
echo "   tail -f training/logs/male_curriculum.log"
echo "   tail -f training/logs/female_curriculum.log"
echo ""
echo "🛑 To stop training:"
echo "   kill $MALE_PID $FEMALE_PID"
echo "=========================================="

# Save PIDs for later reference
echo "$MALE_PID" > training/logs/male_pid.txt
echo "$FEMALE_PID" > training/logs/female_pid.txt
