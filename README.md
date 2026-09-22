# robot_arm_data

Research data about robot manipulators, scraped from public online sources.

![Robot Arm Comparison](robot_arm_summary.png)

## Dataset Overview

This dataset contains specifications for hundreds of robot arms across 4 categories (`collaborative`, `industrial`, `research`, `hobby`). Regenerate the three summary figures with `pixi run plot`.

### Summary plot (`robot_arm_summary.png`)

- **Reach vs Payload Factor** — Payload factor is `payload / robot mass`. Higher values indicate more efficient designs that can lift more relative to their own weight.
- **Color by Type** — Collaborative (green), industrial (red), hobby (blue), research (purple).
- **Convex Hulls** — Shaded regions use **every** arm in that type with mass, payload, and reach (~500 rows). Price is **not** required for hull membership.
- **Dots** — Placed only when **price and repeatability** are both known (~160 rows). Circle size is the value metric `1 / (repeatability × price)` (larger = better value). Legend counts are `priced-dots / hull-members`.

### High payload-efficiency box (`robot_arm_high_pf.png`)

Zoom on arms with **PF > 0.4** and **reach < 2 m**:

- Includes **every** arm with mass/payload/reach in that box — **price and repeatability are not required** (so unpriced research peaks like **Mico 4** / **Jaco 4** / **LWR III** appear).
- Hulls and dots use that same high-PF set; the purple research region extends up to the highest-PF vertex in the box.
- Yellow dashed vertical marks **adult male arm reach (0.75 m)** from the human reference used in the humanoid comparison figure.

### Human & humanoid comparison (`robot_arm_human_comparison_hulls.png`)

![Robot arms vs humans / humanoids (hulls)](robot_arm_human_comparison_hulls.png)

Compares **serial robot-arm design space** to **human** and **humanoid** arm references on the same axes (reach vs payload factor):

- **Type convex hulls** from every arm with mass, payload, and reach (~500 rows)
- **Dots** only when price + repeatability are available (circle size = value metric)
- **Human** references (yellow ★) and **humanoid** arm estimates (orange ▲)
- Dashed **iso-efficiency curves** `PF = 1/(a · reach)` (smaller `a` = higher payload×reach / mass)
- Landmark labels: **LWR III (1:1)**, **Mico 4** (research), collaborative max, **myCobot 280**

Human/humanoid reference numbers live in `data/human_humanoid_arm_data.csv`. The hull figure applies the humanoid assumptions below at plot time (it does not rewrite that CSV).

### Assumptions (humanoids)

All humanoids are treated as one family (orange) with a shared mass rule:

1. **Arm mass ≈ 5% of whole-robot body mass**  
   Used for G1 / G1 EDU / R1 / GR-1 / Optimus Gen2 / Figure 02 / Apollo / Digit / Atlas.

2. **Published “carry” ratings are whole-robot, close-to-body**  
   For Optimus, Figure, Apollo, Digit, and Atlas we convert to a **single-arm, full-reach** estimate as:
   - ÷2 for two arms sharing the load  
   - ÷2 again because carry is near the torso, not at full reach  
   - → **per-arm full-reach payload ≈ carry / 4**

   | Robot | Published carry | Per-arm full-reach (÷4) |
   |---|---|---|
   | Optimus Gen2 | 20 kg | 5.0 kg |
   | Figure 02 | 20 kg | 5.0 kg |
   | Apollo | 25 kg | 6.25 kg |
   | Digit | 16 kg | 4.0 kg |
   | Atlas | 30 kg sustained | 7.5 kg |

3. **Official one-arm ratings** (Unitree G1 / G1 EDU / R1, Fourier GR-1) are already per-arm; we only apply a **÷2 full-reach derate** (not the bimanual ÷2).

4. **Reach** for the five “leading” platforms is a **pixel estimate** from full-body photos (shoulder → hand / fingertip scaled by published height). Atlas uses a human-proportioned estimate (~0.44 × stature) when a clean arms-down photo was unavailable. BD’s published **2.3 m** figure is treated as whole-body workspace reach, not shoulder→hand arm length.

5. **Humans** use separate anthropometric estimates (not the 5% humanoid rule): continuous full-reach hold payloads and shoulder→fingertip reaches for Child / Woman / Man (**0.75 m**) / Strong.

6. These humanoid points are **illustrative**, not datasheet arm specs. Payload factor above ~1 is rare for production serial arms (DLR **LWR III** is the classic research 1:1; Kinova arms are classified here as **research** for separation from cobot fleets — **Mico 4** / **Jaco 4** exceed PF 1 mainly on mid-range continuous ratings with light structures).

### Assumptions (robot hulls)

- Payload factor = `Payload_kg / Weight_kg` using the CSV values as published (no carry÷4 derate on industrial/cobot rows).
- Hull vertices are the convex hull of each `Type` in reach–PF space after dropping rows missing reach, mass, or payload.
- On the summary and humanoid-comparison figures, dots still require price + repeatability; the high-PF zoom does **not**.
- Dashed curves on the humanoid figure are a simple geometric-scaling sketch (`PF ∝ 1/reach`), not a fitted model.

```bash
pixi run plot
```

Outputs: `robot_arm_summary.png`, `robot_arm_high_pf.png`, `robot_arm_human_comparison_hulls.png`.

## Data Sources

All data is collected from corporate websites, company catalogs, resellers, or random blog posts. I tried to make notes of strange sources in the additional notes column when necessary. Occasionally some of the estimates were in euros, so the conversion to dollars may be out of date.

Updated info is welcome via pull request.

## Running the Analysis App

This repo includes an interactive Streamlit app for exploring robot arm data, comparing payload factors, and estimating costs.

### Prerequisites

Install [pixi](https://pixi.sh) if you haven't already:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Running the App

From the project directory, run:

```bash
pixi run app
```

This will start a local Streamlit server and open the app in your browser at `http://localhost:8501`.

### Features

- **Interactive filtering** by robot type (Articulated, Collaborative, Delta, SCARA, etc.)
- **Parameter sliders** for DOF, reach, payload, and repeatability
- **Auto-estimation** of robot mass and cost using regression models trained on the dataset
- **Payload factor visualization** comparing your configuration against real robots
- **Cost analysis** with price-per-DOF calculations

### Regenerating plots

```bash
pixi run plot
```

This runs `generate_summary_plot.py` and writes the three arm figures described above (summary, high-PF zoom with Man-reach line, human/humanoid hull comparison).

## Joint modules

The same repo also tracks **complete robot joints** (motor + gear + encoder, usually with a driver) in `data/robot_joint_data.csv`.

![Joint price fit](robot_joint_price_fit.png)

### Dataset

**142 modules** from 19 manufacturers, in six classes: harmonic (67), QDD (31), series-elastic (21), planetary (9), hobby-servo (8), cycloidal (6).

`Cost_USD` is a one-off street or quote price. `Cost_Flag` says how it was filled:

- **listed** — shop, Alibaba sample, or official list price
- **user** — Maxon HDT ~$5k and HEJ $7.5k quotes
- **estimate** — analog / brand-ladder fill for quote-only catalog parts (Harmonic Drive FHA/SHA, Leaderdrive, Nidec, HIWIN, leftover eRob, etc.)

Empty spec cells stay empty when a datasheet does not publish them. Notes cite the source.

### Price estimator

The joint app fits a two-stage log-price model on **listed + user rows only** (97 training points). Estimates are shown in the CSV but are not used to train.

- Stage 1: `log(price) = a[type] + b·log(torque) + c·log(1/accuracy)`
- Stage 2: weakly regularized residual on speed, size, mass, encoder, brake, driver
- Accuracy is a class default (harmonic 0.25 arcmin, QDD 12, etc.) unless you override it
- Current fit: in-sample R²(log) **0.77**, leave-one-out R² **0.69**, LOO MAE **~$950**
- Torque elasticity is shallow (~0.19): class (QDD vs harmonic vs SEA) moves price more than Nm
- Example: 20 Nm QDD, 90 mm, 0.6 kg → **~$316**. A 50 Nm harmonic in a similar envelope → **~$2.1k**

```bash
pixi run joint-app
pixi run joint-analyze
pixi run python analyze_joint_prices.py --torque 20 --speed 120 --od 90 --mass 0.6 --type qdd
```

`joint-app` is a Streamlit estimator with the same dark theme as the arm app. `joint-analyze` reprints the fit and regenerates `robot_joint_price_fit.png`. Do not insert a bare `--` between the pixi task and argparse flags.

## BLDC motors

Bare brushless motors (not geared joint modules) live in `data/bldc_motor_data.csv`.

![Motor mass fit](robot_motor_mass_fit.png)

### Dataset

Rows span **small → medium → large** BLDC motors from CubeMars (frameless RI/RO + GL gimbal), Maxon (EC / ECX SPEED / ECX FLAT), Faulhaber (micro through 32 mm), Kollmorgen (AKM2G LV), Teknic ClearPath (integrated servo), ODrive (outrunner / hub), T-Motor (Antigravity), plus **AliExpress/Alibaba OEM** coverage (Flipsky esk8 outrunners, HXC/MOSRAC frameless torque motors, scooter/QS hubs). Dense coverage below **1 Nm** (Faulhaber/Maxon micros + CubeMars GL30–40 + MN4006/5006). Core fields:

- **Max_Torque_Nm** — peak / stall / short-term max (primary torque for the mass model)
- **Max_Speed_rpm** — no-load or catalog max speed
- **Weight_kg** — published motor mass (`Weight_Flag=listed`, or `estimate` when only a class analog exists)
- Optional: continuous torque, OD/length, Kv, pole pairs, voltage, street price

Notes cite the catalog / shop page. Empty cells mean the source did not publish that number.

### Mass model

`analyze_motor_mass.py` fits mass from peak torque and max speed on **listed-mass** rows:

```
log(mass) = a + b · log(τ_max) + c · log(ω_max)
```

Physically this is a power-law `mass ∝ τ^b · ω^c`. On the current set, **torque dominates** (`b ≈ 0.72`); the speed exponent is near zero. Current fit (listed-mass rows, n≈76): in-sample R²(log) **~0.88**, leave-one-out R² **~0.87**, LOO MAE **~0.57 kg**. Example: 2 Nm peak @ 4000 rpm → **~0.5 kg**.

```bash
pixi run motor-analyze
pixi run python analyze_motor_mass.py --torque 2.0 --speed 4000
```

Regenerates `robot_motor_mass_fit.png` (predicted vs actual) and `robot_motor_mass_contour.png` (torque–speed map with mass as color contours + measured motors as dots), and prints a one-shot mass estimate when `--torque` / `--speed` are given.

## Gearboxes / reducers

Bare reducers (not full joint modules) live in `data/gearbox_data.csv`.

![Gearbox mass fit](robot_gearbox_mass_fit.png)

### Dataset

**~146 reducers** spanning ratios **~4:1–100:1** (a few up to 111:1), sized so **rated input torque** overlaps the motor set (~0.01–40 Nm input; ~94 rows in the 0.5–30 Nm motor-matched band).

| Type | Sources | Notes |
|---|---|---|
| **harmonic** | Harmonic Drive CSF-2UH / LW; AliExpress HBK/SHF | Mass fixed per frame across 30/50/80/100 |
| **cycloidal** | Nabtesco RV-E | Heavy, high torque; mass per frame |
| **planetary** | Neugart PLE; Maxon GPX; AliExpress FLE42/GP42 | 1-stage ≈ QDD (5–10); 2–3 stage for 25–100 |
| **worm** | NMRV 030–075 | Mass fixed per size; ratios 7.5–100 |
| **spur** | Small industrial / Boston-class analogs | Thin coverage |

Core fields: `Ratio`, `Stages`, `Rated_Output_Torque_Nm`, `Weight_kg`, `OD_mm`, `Length_mm`, derived `Rated_Input_Torque_Nm ≈ T_out / ratio`. Price filled where street listings were clear.

### Mass / size model

Important empirical result: **within a frame, mass barely changes with ratio**. Ratio alone is a weak predictor; **output torque capacity (frame size) and type** dominate:

```
log(mass) = a[type] + b · log(T_out) + c · log(ratio)
```

Typical fit on this set: **b ≈ 0.70**, **c ≈ −0.02**, R²(log) **~0.92**. Per-type mass∝T^b: harmonic ~0.61, planetary ~0.76, worm ~0.57, cycloidal ~0.65. Mass∝ratio alone has R² ≈ 0 for harmonic/worm.

```bash
pixi run gearbox-analyze
pixi run python analyze_gearbox_mass.py --ratio 50 --torque-out 40 --type harmonic
```

Writes `robot_gearbox_mass_fit.png` (mass vs ratio + mass vs torque by type) and `robot_gearbox_od_fit.png` (housing OD vs torque).
