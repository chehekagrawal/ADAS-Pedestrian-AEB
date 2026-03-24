# Human-Aware Safety Modeling in ADAS: Integrating Driver Monitoring with Automatic Emergency Braking

## Abstract
Traditional Advanced Driver Assistance Systems (ADAS) primarily trigger Automatic Emergency Braking (AEB) based on environmental risk factors, most notably the objective Time-To-Collision (TTC) with forward obstacles. However, real-world driving safety is fundamentally a function of both environmental risk and human cognitive state. This report details the extension of an existing vision-based AEB pipeline to incorporate **Human-in-the-Loop Safety Modeling**. By continuously monitoring the driver's facial landmarks and extracting a biomechanical fatigue metric—the Eye Aspect Ratio (EAR)—the system dynamically scales the AEB activation threshold. This creates a novel safety paradigm where the vehicle intervenes earlier for a drowsy driver, transitioning the architecture from a standard computer vision perception task to an Intelligent Safety System prioritizing human capability.

## 1. Methodology: Biomechanical Fatigue Detection
To prevent over-reliance on opaque neural networks for safety-critical states, we employed a highly explainable, geometrically rigorous approach using the `dlib` 68-point facial landmark predictor. This model represents the gold standard in academic literature for facial alignment.

### 1.1 Eye Aspect Ratio (EAR) Formulation
Following Soukupová and Čech (2016), we isolate the landmarks for the left eye (points 36–41) and the right eye (points 42–47). The EAR is calculated dynamically per frame:

$$ EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||} $$

This metric holds the distinct advantage of being invariant to the driver's head pose and the distance to the camera, making it exceptionally robust for in-cabin webcam environments.

### 1.2 Temporal State Classification
A single blink does not indicate drowsiness. Therefore, we implemented a temporal tracking engine that records contiguous frames where $EAR < 0.18$ (Eyes Closed). This engine maps durations directly to classified alertness states, which correspond to literature-backed estimates for human reaction delay:

| State | Temporal Condition | Defined Reaction Delay ($t_r$) |
| :--- | :--- | :--- |
| **ALERT** | Baseline / Rapid Blinks Only | 0.7 s |
| **TIRED** | Frequent blinks / Extended closure ($>0.5$s) | 1.1 s |
| **DROWSY** | Prolonged closure indicating fatigue | 1.6 s |
| **MICROSLEEP** | Unresponsive closure ($>1.5$s) | ∞ (Immediate Trigger) |

## 2. Integration: Adaptive Braking Logic
The core contribution of this module is the mathematical integration of the human state into the vehicle's control matrix.

### 2.1 Standard AEB Trigger
In standard ADAS, the AEB triggers when the Time-To-Collision falls below a set safety threshold ($T_{base\_safe}$), representing the minimum time required for the mechanical brakes to stop the vehicle.
$$ Trigger_{Standard} = TTC < T_{base\_safe} $$

### 2.2 Adaptive Human-Aware Trigger
Our system modifies this equation to absorb the driver's estimated reaction delay ($t_r$). If the driver is drowsy, their reaction delay increases. To compensate safely, the vehicle must "steal" that time back by initiating braking earlier at a higher TTC threshold:
$$ Trigger_{Adaptive} = TTC < (T_{base\_safe} + t_r) $$

#### Example Scenario
Assume $T_{base\_safe} = 1.5$ seconds.
*   **Alert Driver:** $t_r = 0.0$ (relative to baseline). Engine triggers automatically if $TTC < 1.5s$. The driver is given maximum agency to brake normally.
*   **Drowsy Driver:** $t_r = 1.6s$. Engine triggers automatically if $TTC < 3.1s$. The vehicle assumes the driver is compromised and intervenes dramatically earlier, preventing a collision.
*   **Microsleep:** The system bypasses physics calculations and immediately overrides the throttle, triggering full AEB.

## 3. Conclusion and Future Work
By fusing internal cabin perception (driver monitoring) with external environmental tracking (pedestrian trajectories), we construct a holistic risk profile:
$$ Risk = f(Environment\ Risk,\ Driver\ Attention) $$

This methodology is highly publishable as it acknowledges that autonomy must dynamically adapt to human deficiency rather than treating driver models as static constants. Future iterations could strengthen the EAR logic by incorporating Head Pose tracking (detecting distracted gaze) and adjusting thresholds continuously rather than discretely.
