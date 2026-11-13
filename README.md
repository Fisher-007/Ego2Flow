<div align="center">

<h1>Ego2Flow</h1>
<h3>Benchmarking the Traffic-System Impact of Autonomous Driving<br>on Maps Distilled from Real-World Roads</h3>

<p>
<a href="">arXiv</a> ｜
<a href="">Map Data</a> ｜
<a href="https://github.com/Fisher-007/Ego2Flow-Distilled-Map-Generation">Map Generation Code</a>
</p>

</div>

<p align="center">
  <img src="assets/overview.png" alt="Ego2Flow overview: vehicle interactions can propagate speed disturbances; six real-world city maps support sustained traffic, bottleneck response, and perturbation propagation tasks in homogeneous and mixed deployments, evaluated through efficiency, safety, stability, and energy consumption." width="100%">
</p>

## Overview

Autonomous driving is commonly evaluated through an ego vehicle's safety and ability to complete its assigned task. Urban deployment also raises a broader question: **how does a driving policy affect the traffic it joins?**

Driving decisions influence surrounding vehicles. A braking maneuver can induce speed disturbances that propagate through a platoon, delay following vehicles, and affect traffic stability and energy consumption. These collective effects are not fully captured by ego-vehicle performance alone.

**Ego2Flow** is a closed-loop benchmark for evaluating the system-level impact of end-to-end autonomous driving policies on maps distilled from real-world roads. It assesses policy performance across four dimensions: traffic efficiency, safety, stability, and estimated energy consumption.

> Benchmark code and map data are being organized for release.