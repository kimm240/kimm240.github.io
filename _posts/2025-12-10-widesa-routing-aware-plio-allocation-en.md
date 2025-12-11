---
title: '[Paper Review]WideSA: Routing-Aware PLIO Allocation Algorithm'
date: 2025-12-10
permalink: /posts/2025/12/widesa-routing-aware-plio-allocation-en/
excerpt: 'WideSA uses a routing-aware PLIO allocation algorithm to solve routing problems that occur during high AIE utilization. Through this algorithm, it constructs data input/output paths between PLIO ports and AIE cores, improving compilation success rate.'
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

WideSA[^1] uses a routing-aware PLIO allocation algorithm to solve routing problems that occur during high AIE utilization. The goal of this algorithm is to generate placement and routing constraints for constructing data input/output paths between PLIO ports and AIE cores, thereby improving compilation success rate.

## 1. Routing Congestion Analysis and Constraint Formulation

### A. Roles of $p$ and $x$ and NoC Structure

- $x$ (AIE Core): A core that performs computations within the AIE array.
- $p$ (PLIO Port): An I/O port responsible for data communication between PL (Programmable Logic) and the AIE array.
- PLIO is always located at Row 0 of the AIE array.

Since NoC has a mesh structure and PLIO is located at Row 0, the routing problem is transformed into a problem of deciding which column of the AIE array to assign PLIO ports to.

### B. West Direction Congestion ($Cong_{i}^{west}$) and East Direction Congestion ($Cong_{i}^{east}$) Calculation

The algorithm measures routing congestion by counting horizontal data transfers, calculating the total number of data streams ($Cong_{i}^{west}$) that pass through column $i$ boundary in the west direction.

$$\text{Congwest}_i = \sum_{p \in PLIOs, x \in AIEs} W_i[p][x]$$

$W_i[p][x]$ becomes $1$ when the connection between $p$ and $x$ crosses column $i$ boundary in the west direction.

![AI Engine ROUTING](/images/AIE_ROUTING.png)

This figure shows the communication process from the red rectangle (PLIO) at the bottom to the red rectangle (AIE) at the top. At this time, we can see that the red arrows (communication paths from PLIO to AIE) cross the blue column. At this point, the west direction congestion of the blue column increases.

The east direction can be understood as the opposite of the west direction.

### C. Routing Constraints

The PLIO allocation problem is formulated as a satisfiability problem to ensure that routing congestion does not exceed available resources.

$$\forall i \in \text{Columns}, \text{Congwest}_i \leq RC_{west}, \text{Congeast}_i \leq RC_{east}$$

$RC_{west}$ and $RC_{east}$ are the available routing resources in the AIE array.

The congestion calculated for all columns must not exceed the maximum routing resources provided by the hardware, and finding a PLIO placement that satisfies this constraint is key.

## 2. Heuristic Greedy Algorithm

To find PLIO allocations that satisfy the constraints, WideSA uses a heuristic greedy algorithm.

![WIDESA_ALGO](/images/WIDESA_ALGO.png)

Initialize available placement set $A$ as all columns where PLIO ports can be placed.
Collect column coordinates ($x_{col}$) of all AIE cores connected to the current PLIO to be allocated into list $S$.
Sort list $S$ to calculate the median ($S[num/2]$).
Find the nearest available coordinate to the median position from available placement set $A$ and assign it to the current PLIO ($P[i]$).
Remove the assigned coordinate from $A$ so it is not used for the next PLIO allocation.

By placing PLIO near the median, the average horizontal distance that data must travel through NoC can be minimized. This minimizes routing congestion.

---

**Language**: [한국어 (Korean)](/posts/2025/12/widesa-routing-aware-plio-allocation/)

[^1]: T. Dai, B. Shi and G. Luo, "WideSA: A High Array Utilization Mapping Scheme for Uniform Recurrences on ACAP," 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), Valencia, Spain, 2024, pp. 1-6

