# Assistant evaluation, merged (606 examples per system)

| System | Tool match | Tool match (tool rows) | Args valid | Language ok | hi ok | hinglish ok | en ok | Safety flags/ex | Latency s |
|---|---|---|---|---|---|---|---|---|---|
| rules | 77.2% | 46.5% | - | 39.1% | 4.1% | 0.0% | 100.0% | 0.02 | 0.02 |
| ollama:krishidisha-v1 | 74.3% | 1.3% | - | 85.5% | 88.4% | 68.8% | 99.6% | 0.03 | 10.29 |
| ollama:krishidisha-v2 | 57.8% | 86.6% | - | 66.0% | 76.2% | 23.1% | 100.0% | 0.06 | 29.58 |

Tool rows = eval examples whose reference trajectory calls at least one tool (157 of them).
