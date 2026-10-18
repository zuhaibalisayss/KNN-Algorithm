// Deterministic datasets for the KNN presentation
import { DataPoint } from '../utils/knn';

/**
 * Main interactive dataset.
 * Two classes positioned so that changing K produces different predictions
 * when the new observation is near the class boundary.
 * 
 * Class A (blue circles): clustered in upper-left region
 * Class B (terracotta squares): clustered in lower-right region
 * Some overlap near the diagonal boundary.
 */
export const mainDataset: DataPoint[] = [
  // Class A - upper-left cluster
  { id: 'A1', x: 2, y: 8, label: 'A' },
  { id: 'A2', x: 3, y: 7, label: 'A' },
  { id: 'A3', x: 2.5, y: 9, label: 'A' },
  { id: 'A4', x: 4, y: 8.5, label: 'A' },
  { id: 'A5', x: 1.5, y: 7.5, label: 'A' },
  { id: 'A6', x: 3.5, y: 6.5, label: 'A' },
  { id: 'A7', x: 5, y: 7, label: 'A' },
  { id: 'A8', x: 2, y: 6, label: 'A' },
  { id: 'A9', x: 4.5, y: 9, label: 'A' },
  { id: 'A10', x: 3, y: 8, label: 'A' },
  // Class B - lower-right cluster
  { id: 'B1', x: 7, y: 2, label: 'B' },
  { id: 'B2', x: 8, y: 3, label: 'B' },
  { id: 'B3', x: 7.5, y: 1.5, label: 'B' },
  { id: 'B4', x: 9, y: 2.5, label: 'B' },
  { id: 'B5', x: 6.5, y: 3.5, label: 'B' },
  { id: 'B6', x: 8.5, y: 4, label: 'B' },
  { id: 'B7', x: 7, y: 4.5, label: 'B' },
  { id: 'B8', x: 8, y: 1, label: 'B' },
  { id: 'B9', x: 6, y: 2.5, label: 'B' },
  { id: 'B10', x: 9, y: 3.5, label: 'B' },
  // Boundary points - these create interesting K-dependent behavior
  { id: 'A11', x: 5.5, y: 5.5, label: 'A' },
  { id: 'B11', x: 5, y: 4, label: 'B' },
  { id: 'A12', x: 4.5, y: 5, label: 'A' },
  { id: 'B12', x: 6, y: 4.5, label: 'B' },
];

/**
 * Default position for the new observation.
 * Placed near the boundary so changing K changes the prediction.
 * At K=3: nearest are A11(5.5,5.5), B11(5,4), A12(4.5,5) → 2A, 1B → Class A
 * At K=5: adds B12(6,4.5) and possibly another → could change
 */
export const defaultNewPoint = { x: 5.2, y: 5.0 };

/**
 * Regression dataset for the classification vs regression section.
 * Points with numerical target values (e.g., house prices in thousands).
 */
export const regressionDataset = [
  { x: 2, y: 2, target: 150 },
  { x: 3, y: 3, target: 180 },
  { x: 4, y: 2.5, target: 200 },
  { x: 2.5, y: 4, target: 220 },
  { x: 5, y: 5, target: 280 },
  { x: 6, y: 4, target: 310 },
  { x: 7, y: 6, target: 350 },
  { x: 8, y: 7, target: 400 },
  { x: 3, y: 6, target: 260 },
  { x: 5.5, y: 3, target: 240 },
  { x: 7, y: 3, target: 270 },
  { x: 4, y: 7, target: 300 },
];

export const defaultRegressionPoint = { x: 5, y: 4.5 };

/**
 * Small dataset for the step-by-step algorithm walkthrough.
 */
export const walkthroughDataset: DataPoint[] = [
  { id: 'W1', x: 2, y: 3, label: 'A' },
  { id: 'W2', x: 3, y: 2, label: 'A' },
  { id: 'W3', x: 2.5, y: 4, label: 'A' },
  { id: 'W4', x: 7, y: 7, label: 'B' },
  { id: 'W5', x: 8, y: 6, label: 'B' },
  { id: 'W6', x: 7.5, y: 8, label: 'B' },
  { id: 'W7', x: 5, y: 5, label: 'A' },
  { id: 'W8', x: 6, y: 6, label: 'B' },
];

export const walkthroughNewPoint = { x: 4.5, y: 4.5 };

/**
 * Dataset for K comparison visualization.
 * Designed to show clear differences between K=1, K=5, K=15.
 */
export const kComparisonDataset: DataPoint[] = [
  // Class A cluster - left side
  { id: 'K1', x: 1, y: 5, label: 'A' },
  { id: 'K2', x: 2, y: 6, label: 'A' },
  { id: 'K3', x: 1.5, y: 4, label: 'A' },
  { id: 'K4', x: 3, y: 5.5, label: 'A' },
  { id: 'K5', x: 2.5, y: 7, label: 'A' },
  { id: 'K6', x: 1, y: 7, label: 'A' },
  { id: 'K7', x: 3.5, y: 4, label: 'A' },
  { id: 'K8', x: 2, y: 3, label: 'A' },
  // Class B cluster - right side
  { id: 'K9', x: 7, y: 5, label: 'B' },
  { id: 'K10', x: 8, y: 6, label: 'B' },
  { id: 'K11', x: 7.5, y: 4, label: 'B' },
  { id: 'K12', x: 9, y: 5.5, label: 'B' },
  { id: 'K13', x: 8.5, y: 7, label: 'B' },
  { id: 'K14', x: 6.5, y: 7, label: 'B' },
  { id: 'K15', x: 9, y: 3, label: 'B' },
  { id: 'K16', x: 7, y: 3, label: 'B' },
  // Scattered noise points to show K=1 sensitivity
  { id: 'K17', x: 5, y: 2, label: 'A' },  // noise - A in B territory
  { id: 'K18', x: 5, y: 8, label: 'B' },  // noise - B in A territory
];
