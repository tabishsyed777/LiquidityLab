# LiquidityLab

**Created by Tabish Syed**

LiquidityLab is an intraday payment-flow forecasting and liquidity-buffer simulator. It compares a fixed historical reserve with a forecast-led reserve under configurable stress scenarios, using entirely synthetic flow data.

The goal is to make the modeling and operating assumptions visible: time-based validation, forecast error, settlement-window demand, reserve requirements, and potential shortfalls.

## Run locally

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots

### Forecast dashboard

![LiquidityLab intraday forecast and reserve policy comparison](assets/liquiditylab-dashboard.png)

### Stress-test replay

![LiquidityLab stress-test reserve comparison](assets/liquiditylab-stress-test.png)

## Explore the simulation

The default dataset contains 180 simulated days with 26 settlement windows per day. A gradient-boosting model forecasts net outflow for each window. The final period is withheld for a chronological evaluation.

Choose an outflow and inflow stress factor to compare:

- **Fixed reserve:** 90th percentile of historical peak cumulative net outflow.
- **Forecast reserve:** peak predicted cumulative net outflow plus a calibration error buffer from a separate time segment.

The dashboard reports average reserve, days with shortfall, and shortfall windows. These are simulation outputs, not a recommended balance for a real institution.

## Score a CSV

Upload daily window totals with these columns:

| Column | Meaning |
| --- | --- |
| `date` | Business or calendar date |
| `slot` | Ordered settlement window number, starting at 0 |
| `incoming_m` | Incoming payment value, in millions of arbitrary currency units |
| `outgoing_m` | Outgoing payment value, in millions of arbitrary currency units |

Provide at least six dates of history. The uploaded flows are scored by a model trained on synthetic data, so units and estimates are not calibrated to a real payment system.

## Method and limits

Features use only prior days for the same settlement window, including recent net flow and incoming/outgoing amounts. Model evaluation is chronological. The reserve comparison estimates liquidity demand from net outflows, but does not model real settlement rules, collateral, intraday credit, payment priorities, funding costs, or counterparty behavior.

This is an educational prototype, not financial advice, treasury guidance, or a production liquidity-risk system. It uses no real bank or customer data.

## License

MIT. See [LICENSE](LICENSE).
