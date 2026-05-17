# Project Kijiji � Training Run Log


## Run 20260515_211610
*2026-05-15T21:16:10.099421+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.4167 | -0.0535 | 49 |
| 2 | 0.4252 | 0.0486 | 54 |
| 3 | 0.3269 | -0.0716 | 51 |

**Mean CV val loss**: 0.3896
**Mean CV separation**: -0.0255

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1553 |
| Pos mean score | 2.861 |
| Neg mean score | 2.6466 |
| Separation | 0.2144 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1553 | **77.6% better than random** |

### Peering Simulations
| Proposed Link | Regional Δ | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.03ms) |
| Kinshasa → Lagos | +0.01ms | LOS (+0.06ms) |
| Kinshasa → Nairobi | +0.01ms | NBO (+0.05ms) |

**Training time**: 1.7s

---

## Run 20260517_185139
*2026-05-17T18:51:39.587623+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.4167 | -0.0535 | 49 |
| 2 | 0.4252 | 0.0486 | 54 |
| 3 | 0.3269 | -0.0716 | 51 |

**Mean CV val loss**: 0.3896
**Mean CV separation**: -0.0255

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1553 |
| Pos mean score | 2.861 |
| Neg mean score | 2.6466 |
| Separation | 0.2144 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1553 | **77.6% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.03ms) |
| Kinshasa → Lagos | +0.01ms | LOS (+0.06ms) |
| Kinshasa → Nairobi | +0.01ms | NBO (+0.05ms) |

**Training time**: 6.5s

---

## Run 20260517_185429
*2026-05-17T18:54:29.798013+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.3977 | 0.0010 | 44 |
| 2 | 0.4379 | 0.0029 | 39 |
| 3 | 0.3035 | 0.0117 | 41 |

**Mean CV val loss**: 0.3797
**Mean CV separation**: 0.0052

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1504 |
| Pos mean score | 3.3782 |
| Neg mean score | 3.3793 |
| Separation | -0.0011 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1504 | **78.3% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.01ms) |

**Training time**: 2.4s

---

## Run 20260517_190436
*2026-05-17T19:04:36.326168+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.3977 | 0.0010 | 44 |
| 2 | 0.4379 | 0.0029 | 39 |
| 3 | 0.3035 | 0.0117 | 41 |

**Mean CV val loss**: 0.3797
**Mean CV separation**: 0.0052

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1504 |
| Pos mean score | 3.3782 |
| Neg mean score | 3.3793 |
| Separation | -0.0011 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1504 | **78.3% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.01ms) |

**Training time**: 3.7s

---

## Run 20260517_190505
*2026-05-17T19:05:05.184913+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.3977 | 0.0010 | 44 |
| 2 | 0.4379 | 0.0029 | 39 |
| 3 | 0.3035 | 0.0117 | 41 |

**Mean CV val loss**: 0.3797
**Mean CV separation**: 0.0052

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1504 |
| Pos mean score | 3.3782 |
| Neg mean score | 3.3793 |
| Separation | -0.0011 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1504 | **78.3% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.01ms) |

**Training time**: 1.3s

---

## Run 20260517_194632
*2026-05-17T19:46:32.624028+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 20 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.3977 | 0.0010 | 44 |
| 2 | 0.4379 | 0.0029 | 39 |
| 3 | 0.3035 | 0.0117 | 41 |

**Mean CV val loss**: 0.3797
**Mean CV separation**: 0.0052

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1504 |
| Pos mean score | 3.3782 |
| Neg mean score | 3.3793 |
| Separation | -0.0011 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1504 | **78.3% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.01ms) |

**Training time**: 7.8s

---

## Run 20260517_194756
*2026-05-17T19:47:56.627839+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 30 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.3977 | 0.0010 | 54 |
| 2 | 0.4404 | -0.0040 | 57 |
| 3 | 0.3015 | 0.0104 | 60 |

**Mean CV val loss**: 0.3799
**Mean CV separation**: 0.0025

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1474 |
| Pos mean score | 3.4003 |
| Neg mean score | 3.4134 |
| Separation | -0.0131 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1474 | **78.7% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.01ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.01ms) |

**Training time**: 1.8s

---

## Run 20260517_195546
*2026-05-17T19:55:46.713368+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 30 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.6907 | 0.0003 | 31 |
| 2 | 0.6937 | -0.0004 | 31 |
| 3 | 0.6966 | 0.0004 | 31 |

**Mean CV val loss**: 0.6937
**Mean CV separation**: 0.0001

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.6166 |
| Pos mean score | 1.1487 |
| Neg mean score | 1.1457 |
| Separation | 0.003 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.6166 | **11.0% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.01ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.00ms) |

**Training time**: 1.9s

---

## Run 20260517_200452
*2026-05-17T20:04:52.068843+00:00*

### Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Learning rate | 0.001 |
| Early stopping patience | 30 |
| Hidden dim | 64 |
| Seed | 42 |

### Walk-Forward Cross-Validation (3 folds)
| Fold | Val Loss | Separation | Stopped Epoch |
|------|----------|------------|---------------|
| 1 | 0.6891 | -0.0002 | 31 |
| 2 | 0.6942 | -0.0000 | 31 |
| 3 | 0.7034 | -0.0000 | 31 |

**Mean CV val loss**: 0.6956
**Mean CV separation**: -0.0001

### Test Results (2024-H2 held-out window)
| Metric | Value |
|--------|-------|
| Test loss | 0.1434 |
| Pos mean score | 4.5414 |
| Neg mean score | 4.4918 |
| Separation | 0.0496 |

### Baseline Comparison
| Model | Loss | Improvement |
|-------|------|-------------|
| Random (BCE lower bound) | 0.6931 | — |
| GraphSAGE (ours) | 0.1434 | **79.3% better than random** |

### Peering Simulations
| Proposed Link | Regional Delta (ms) | Top Beneficiary |
|---------------|------------|-----------------|
| Addis Ababa → Nairobi | +0.00ms | NBO (+0.00ms) |
| Kinshasa → Lagos | +0.00ms | LOS (+0.00ms) |
| Kinshasa → Nairobi | +0.00ms | NBO (+0.00ms) |

**Training time**: 1.6s

---
