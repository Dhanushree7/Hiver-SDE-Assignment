# Top 5 Failure Modes

## 1. Hardware damage classified as connectivity
- Example ID: 1874494
- Gold: hardware_accessories
- Predicted: connectivity
- Confidence: 0.2649
- Failure: Repeatedly cracked phones were misclassified as connectivity issues.
- Improvement: Strengthen physical-damage and hardware-component signals.

## 2. Camera hardware issue classified as update issue
- Example ID: 2123864
- Gold: hardware_accessories
- Predicted: ios_update_issues
- Confidence: 0.1904
- Failure: A grainy MacBook camera problem was classified as an update issue.
- Improvement: Add device-specific hardware and camera signals.

## 3. Keyboard issue classified as update issue
- Example ID: 2578723
- Gold: keyboard_input
- Predicted: ios_update_issues
- Confidence: 0.4118
- Failure: An autocorrect problem was overridden by update-related language.
- Improvement: Prioritize explicit keyboard and autocorrect symptoms.

## 4. Purchase issue classified as update issue
- Example ID: 1020880
- Gold: purchases_orders_refunds
- Predicted: ios_update_issues
- Confidence: 0.2208
- Failure: An iBooks download problem was classified as an update issue.
- Improvement: Strengthen purchase, download, and order-related signals.

## 5. Connectivity issue classified as device performance
- Example ID: 333252
- Gold: connectivity
- Predicted: device_performance
- Confidence: 0.4928
- Failure: Wi-Fi and Bluetooth problems were classified as performance issues.
- Improvement: Prioritize Wi-Fi and Bluetooth indicators.

## General Observation
Low-confidence predictions and overlapping issue descriptions remain important sources of classification errors. The agent should use clarification and escalation when the evidence is insufficient.
