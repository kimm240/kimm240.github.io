---
title: 'WideSA의 Routing-Aware PLIO 할당 알고리즘'
date: 2025-12-10
permalink: /posts/2025/12/widesa-routing-aware-plio-allocation/
excerpt: 'WideSA는 Versal ACAP에서 높은 AIE 배열 활용도를 달성하기 위한 매핑 방안입니다. 라우팅-인식 PLIO 할당 알고리즘을 통해 PLIO 포트와 AIE 코어 사이의 데이터 입/출력 경로를 구축하고, 컴파일 성공률을 높입니다.'
tags:
  - AI Accelerator
  - Versal AI Engine
  - WideSA
  - AIE
  - PLIO
  - Routing
  - NoC
categories:
  - Versal AI Engine
---

WideSa[^1]에서는 고도의 AIE 활용 시 발생하는 라우팅 문제를 해결하기 위해 라우팅-인식 PLIO 할당 알고리즘이 사용됩니다. 이 알고리즘의 목표는 PLIO 포트와 AIE 코어 사이에 데이터 입/출력 경로를 구축하기 위한 배치 및 라우팅 제약 조건을 생성하여, 컴파일 성공률을 높이는 것입니다.

## 1. 라우팅 혼잡 분석 및 제약 조건 공식화

### A. $p$와 $x$의 역할 및 NoC 구조

- $x$ (AIE Core): AIE 배열 내에서 연산을 수행하는 코어입니다.
- $p$ (PLIO Port): PL(Programmable Logic)과 AIE 배열 간의 데이터 통신을 담당하는 I/O 포트입니다.
- PLIO는 항상 AIE 배열의 Row 0에 위치합니다.

NoC가 메시(mesh) 구조이고 PLIO가 Row 0에 위치하므로, 라우팅 문제는 PLIO 포트를 AIE 배열의 어느 열(Column)에 할당할지 결정하는 문제로 전환됩니다.

### B. 서쪽 방향 혼잡도 ($Cong_{i}^{west}$). 동쪽 방향 혼잡도($Cong_{i}^{east}$) 계산

알고리즘은 수평 데이터 전송 횟수를 세어 라우팅 혼잡을 측정하며, 특정 열 $i$ 경계를 서쪽(West) 방향으로 통과하는 데이터 스트림의 총 개수($Cong_{i}^{west}$)를 계산합니다. 

$$\text{Congwest}_i = \sum_{p \in PLIOs, x \in AIEs} W_i[p][x]$$

$W_i[p][x]$는 $p$와 $x$ 간의 연결이 열 $i$ 경계를 서쪽으로 가로지를 때 $1$이 됩니다.

![AI Engine ROUTING](/images/AIE_ROUTING.png)

해당 Figure는 아래 빨간색 사각형(PLIO)에서 위 빨간색 사각형(AIE)으로 통신하는 과정을 나타낸  것입니다. 이 때, 빨간색 화살표(PLIO에서 AIE로 향하는 통신 경로)가 파란색 열을 가로지르는 것을 확인할 수 있습니다. 이 때, 파란색 열의 서쪽 방향 혼잡도가 늘어납니다.

동쪽 방향은 서쪽 방향의 반대로 이해할 수 있습니다.

### C. 라우팅 제약 조건 

PLIO 할당 문제는 라우팅 혼잡이 가용 리소스를 초과하지 않도록 하는 충족 가능성 문제로 공식화됩니다.

$$\forall i \in \text{Columns}, \text{Congwest}_i \leq RC_{west}, \text{Congeast}_i \leq RC_{east}$$

$RC_{west}$와 $RC_{east}$는 AIE 배열에서 사용 가능한 라우팅 리소스입니다.

모든 열에서 계산된 혼잡도가 하드웨어에서 제공하는 최대 라우팅 리소스를 초과해서는 안 되며, 이 제약 조건을 만족하는 PLIO 배치를 찾는 것이 핵심입니다.

## 2. 휴리스틱 탐욕 알고리즘 

제약 조건을 만족하는 PLIO 할당을 찾기 위해, WideSA는 휴리스틱 그리디 알고리즘을 사용합니다. 

![WIDESA_ALGO](/images/WIDESA_ALGO.png)

PLIO 포트를 배치할 수 있는 모든 열을 가용 배치 세트 $A$로 초기화합니다.
할당하려는 현재 PLIO에 연결된 모든 AIE 코어의 열 좌표($x_{col}$)를 목록 $S$에 수집합니다.
목록 $S$를 정렬하여 중앙값($S[num/2]$)을 계산합니다.
중앙값 위치에 가장 가까운 사용 가능한 좌표를 가용 배치 세트 $A$에서 찾아 현재 PLIO에 할당합니다 ($P[i]$).
할당된 좌표는 다음 PLIO 할당에 사용되지 않도록 $A$에서 제거됩니다.

PLIO를 중앙값 근처에 배치함으로써, 데이터가 NoC를 통해 이동해야 하는 평균 수평 거리가 최소화할 수 있습니다. 이렇게 함으로써 라우팅 혼잡도를 최소화할 수 있습니다.

---

**Language**: [English](/posts/2025/12/widesa-routing-aware-plio-allocation-en/)

[^1]: T. Dai, B. Shi and G. Luo, "WideSA: A High Array Utilization Mapping Scheme for Uniform Recurrences on ACAP," 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), Valencia, Spain, 2024, pp. 1-6
