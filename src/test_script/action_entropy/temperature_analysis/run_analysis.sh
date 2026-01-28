#!/bin/bash
# Temperature Entropy Analysis 실행 스크립트

echo "========================================"
echo "Temperature vs Entropy Analysis"
echo "========================================"

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# Python 실행
python temperature_entropy_analysis.py

echo ""
echo "분석 완료! results/ 폴더를 확인하세요."
