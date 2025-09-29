# Validation Strategy

## Philosophy

We validate on KNOWN phenomena before applying to UNKNOWN questions.

## Three-Stage Approach

### Stage 1: China Shock (Known Spatial Boundaries)
- **Data**: Autor, Dorn, Hanson (2013)
- **Known result**: ~150km boundary, 0.05 decay rate
- **Test**: Can our method detect these known boundaries?
- **Criterion**: Within 20% of literature values

### Stage 2: Spatial vs Network Clarification
- **Question**: Are these separate mechanisms?
- **Test**: Independence tests, interaction effects
- **Outcome**: Model separately or jointly

### Stage 3: Application (Only if Stages 1-2 Pass)
- **Data**: AI investment (exploratory)
- **Framing**: Application, not validation
- **Limitations**: Fully documented

## Decision Tree

```
Stage 1: China Shock Validation
├── SUCCESS → Proceed to Stage 2
└── FAILURE → Refine method, repeat Stage 1

Stage 2: Spatial vs Network
├── INDEPENDENT → Separate models
├── DEPENDENT → Joint model
└── UNCLEAR → More data needed

Stage 3: Application
├── After Stages 1-2 pass
└── With full limitations documented
```

## Validation Criteria

Method is validated if it:
1. Detects China shock boundaries within ±20% of literature
2. Achieves better out-of-sample prediction than baseline
3. Identifies correct spatial decay patterns
4. Does not produce false positives in placebo tests
