---
title: '[논문 리뷰]MaxEVA의 AI Engine 커널 배치 전략 및 통신 방식'
date: 2025-12-07
permalink: /posts/2025/12/maxeva-ai-engine-kernel-placement/
excerpt: 'MaxEVA 프레임워크는 Versal AI Engine 배열의 활용도를 최대화하고 MatMul 커널과 애더 트리 간의 통신에서 DMA 사용을 최소화하여 효율성을 높이는 정교한 배치 전략을 사용합니다.'
tags:
  - AI Accelerator
  - Versal AI Engine
  - MaxEVA
  - AIE
  - DMA
  - Kernel Placement
categories:
  - Versal AI Engine
---

MaxEVA 프레임워크[^1]는 Versal AI Engine (AIE) 배열의 활용도를 최대화하고 MatMul 커널과 애더 트리(Add kernel) 간의 통신에서 DMA(Direct Memory Access) 사용을 최소화하여 효율성을 높이는 정교한 배치 전략을 사용합니다. 

## 1. 커널 그룹 구성 및 커널 수

![AI Engine 통신 방식](/images/AIE3.png)

MaxEVA의 매핑 체계에서 `Y`는 MatMul 커널 그룹을 정의하는 핵심 매개변수입니다.

- Y 값은 그룹 내 MatMul 커널('×' 기호)의 수를 결정합니다. 예를 들어, Y=4인 그룹에는 4개의 MatMul 커널이 포함되며, 이들은 각각 별도의 AIE 코어에 매핑됩니다.
- 이 그룹은 Y−1개의 Add 커널로 구성된 애더 트리('+' 기호)를 포함하며, 이 애더 트리는 하나의 AIE 코어에 순차적으로 매핑됩니다.

## 2. AIE 배열 내 커널 간 통신 방식

AIE 배열 내에서 데이터 이동 및 커널 간 통신은 주로 두 가지 방식으로 이루어집니다. MaxEVA의 배치 전략은 직접 메모리 공유를 활용하여 DMA 사용을 회피하는 데 초점을 맞춥니다.

![AI Engine 통신 방식](/images/AIE4.png)

| 통신 방식 | 특징 | 
|----------|------|
| 직접 메모리 공유 (Direct Memory Sharing) | 인접한 AIE 코어들이 직접 상대방의 메모리(32KB)에 접근할 수 있는 방식 | 
| DMA (Direct Memory Access) / AXI4-Stream Switch | 비인접 AIE 코어 간의 통신에 사용되며, 프로그래머블 스위치를 통한 DMA 메커니즘을 이용합니다. 직접 접근 방식보다 통신 지연 시간이 증가하고 더 많은 메모리 리소스를 요구합니다. |


## 3. DMA 사용 회피를 위한 커널 배치 기법


MaxEVA는 Y개의 MatMul 커널 그룹을 배치할 때, 애더 트리 코어가 인접 MatMul 커널의 출력 버퍼에 직접 접근할 수 있도록 설계합니다.


![AI Engine 통신 방식](/images/AIE1.png)

### 배치 조정 예시 (0, 0 그룹)

Y=4 그룹의 예시에서, 애더 트리 코어는 4개의 MatMul 커널 중 3개의 메모리에 직접 접근할 수 있습니다. (1, 0) 위치의 커널처럼 직접 접근이 어려운 경우, 해당 MatMul 커널의 출력 버퍼는 북쪽 위치인 (1, 1)에 배치될 수 있으며, 이 위치는 애더 트리가 직접 접근 가능한 위치이므로 DMA 사용을 방지합니다.

### 행 기반 접근 예시 (0, 5 그룹)

AIE의 메모리 접근은 행에 따라 달라지기 때문에 (짝수 행은 서쪽, 홀수 행은 동쪽 접근), (0, 5) 그룹의 애더 트리는 4개 중 2개의 커널에만 직접 접근 가능합니다. 나머지 커널인 (0, 5)와 (0, 7)의 출력 버퍼는 애더 트리가 직접 접근 가능한 (1, 5)와 (1, 7) 위치에 배치되어 DMA를 회피합니다.

## 4. AIE 배열 전체를 채우는 배치 패턴 P1 및 P2

MaxEVA는 3에서 제시한 커널 배치 기법을 활용하여, 두 가지 패턴을 제안합니다.

![AI Engine 통신 방식](/images/AIE2.png)

- 패턴 P1: AIE 배열을 97.5% 이상 채우기 위해 "T"자 모양과 유사한 형태를 사용하며, 이로 인해 하나의 MatMul 출력 버퍼에서 약간의 DMA 사용이 발생할 수 있습니다.
- 패턴 P2: DMA 사용을 완전히 피하도록 설계된 형태로만 구성되어, DMA 사용이 전혀 발생하지 않습니다.

---

**Language**: [English](/posts/2025/12/maxeva-ai-engine-kernel-placement-en/)

[^1]: E. Taka, A. Arora, K. -C. Wu and D. Marculescu, "MaxEVA: Maximizing the Efficiency of Matrix Multiplication on Versal AI Engine,"International Conference on Field Programmable Technology (ICFPT), Yokohama, Japan, 2023, pp. 96-105,