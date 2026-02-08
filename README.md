# robot_arm_data

Research data about robot manipulators, scraped from public online sources.

![Robot Arm Comparison](robot_arm_summary.png)

## Dataset Overview

This dataset contains specifications for **127+ robot arms** across 4 categories with complete data (payload, reach, mass, repeatability, and price). The visualization above shows:

- **Reach vs Payload Factor** — Payload factor is the ratio of payload capacity to robot mass. Higher values indicate more efficient designs that can lift more relative to their own weight.
- **Color by Type** — Collaborative robots (green) cluster at shorter reaches with higher payload efficiency. Industrial robots (red) span wider reach ranges but with lower efficiency. Hobby (blue) and Research (purple) robots occupy smaller niches.
- **Circle Size = Value Metric** — Calculated as `1 / (repeatability × price)`. Larger circles represent better value: higher precision at lower cost.
- **Convex Hulls** — Shaded regions show the design space each robot type occupies, highlighting where categories overlap and compete.

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

### Regenerating the Summary Plot

To regenerate the `robot_arm_summary.png` image:

```bash
pixi run plot
```
