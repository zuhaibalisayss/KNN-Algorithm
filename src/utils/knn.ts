// KNN Algorithm Implementation
// This module contains pure functions for KNN calculations.

export type DistanceMetric = 'euclidean' | 'manhattan';
export type ClassLabel = 'A' | 'B';

export interface DataPoint {
  id: string;
  x: number;
  y: number;
  label: ClassLabel;
}

export interface PredictionResult {
  k: number;
  metric: DistanceMetric;
  neighbours: NeighbourInfo[];
  votes: { A: number; B: number };
  predictedClass: ClassLabel | 'tie';
  voteShare: number; // 0-1, fraction of votes for winning class
  confidence: string; // formatted percentage
}

export interface NeighbourInfo {
  id: string;
  x: number;
  y: number;
  label: ClassLabel;
  distance: number;
  isWinningClass: boolean;
}

/**
 * Calculate Euclidean distance between two 2D points.
 * d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
 */
export function euclideanDistance(
  x1: number, y1: number,
  x2: number, y2: number
): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

/**
 * Calculate Manhattan distance between two 2D points.
 * d = |x2 - x1| + |y2 - y1|
 */
export function manhattanDistance(
  x1: number, y1: number,
  x2: number, y2: number
): number {
  return Math.abs(x2 - x1) + Math.abs(y2 - y1);
}

/**
 * Calculate distance using the specified metric.
 */
export function calculateDistance(
  x1: number, y1: number,
  x2: number, y2: number,
  metric: DistanceMetric
): number {
  if (metric === 'manhattan') {
    return manhattanDistance(x1, y1, x2, y2);
  }
  return euclideanDistance(x1, y1, x2, y2);
}

/**
 * Run the complete KNN classification algorithm.
 * 
 * Steps:
 * 1. Calculate distances from newPoint to all training points
 * 2. Sort by distance (ascending), with deterministic tie-breaking by id
 * 3. Select K nearest neighbours
 * 4. Count votes per class
 * 5. Determine winner or tie
 */
export function knnClassify(
  trainingData: DataPoint[],
  newPointX: number,
  newPointY: number,
  k: number,
  metric: DistanceMetric
): PredictionResult {
  // Clamp K to dataset size
  const effectiveK = Math.min(k, trainingData.length);

  // Step 1 & 2: Calculate distances and sort
  const distances = trainingData.map(point => ({
    ...point,
    distance: calculateDistance(newPointX, newPointY, point.x, point.y, metric)
  }));

  // Sort by distance ascending, tie-break by id for determinism
  distances.sort((a, b) => {
    if (Math.abs(a.distance - b.distance) < 1e-10) {
      return a.id.localeCompare(b.id);
    }
    return a.distance - b.distance;
  });

  // Step 3: Select K nearest
  const neighbours = distances.slice(0, effectiveK);

  // Step 4: Count votes
  const votes = { A: 0, B: 0 };
  neighbours.forEach(n => {
    votes[n.label]++;
  });

  // Step 5: Determine winner
  let predictedClass: ClassLabel | 'tie';
  let voteShare: number;

  if (votes.A === votes.B) {
    predictedClass = 'tie';
    voteShare = votes.A / effectiveK; // equal share
  } else if (votes.A > votes.B) {
    predictedClass = 'A';
    voteShare = votes.A / effectiveK;
  } else {
    predictedClass = 'B';
    voteShare = votes.B / effectiveK;
  }

  // Build neighbour info with winning class flag
  const neighbourInfo: NeighbourInfo[] = neighbours.map(n => ({
    id: n.id,
    x: n.x,
    y: n.y,
    label: n.label,
    distance: n.distance,
    isWinningClass: n.label === predictedClass
  }));

  const confidence = (voteShare * 100).toFixed(1) + '%';

  return {
    k: effectiveK,
    metric,
    neighbours: neighbourInfo,
    votes,
    predictedClass,
    voteShare,
    confidence
  };
}

/**
 * KNN Regression: predict the mean of K nearest neighbours' target values.
 */
export function knnRegress(
  trainingData: { x: number; y: number; target: number }[],
  newPointX: number,
  newPointY: number,
  k: number,
  metric: DistanceMetric
): { predictedValue: number; neighbours: { x: number; y: number; target: number; distance: number }[] } {
  const effectiveK = Math.min(k, trainingData.length);

  const distances = trainingData.map(point => ({
    ...point,
    distance: calculateDistance(newPointX, newPointY, point.x, point.y, metric)
  }));

  distances.sort((a, b) => a.distance - b.distance);

  const neighbours = distances.slice(0, effectiveK);
  const predictedValue = neighbours.reduce((sum, n) => sum + n.target, 0) / effectiveK;

  return { predictedValue, neighbours };
}

/**
 * Generate a text explanation of the KNN result.
 */
export function generateExplanation(result: PredictionResult): string {
  const { k, metric, votes, predictedClass, confidence } = result;
  
  let explanation = `K = ${k}. The algorithm selected the ${k} closest observation${k > 1 ? 's' : ''} using ${metric} distance. `;
  explanation += `${votes.A} neighbour${votes.A !== 1 ? 's' : ''} belong${votes.A === 1 ? 's' : ''} to Class A, and ${votes.B} belong${votes.B !== 1 ? 's' : ''} to Class B. `;
  
  if (predictedClass === 'tie') {
    explanation += `The votes are equal, resulting in a tie. No single class has majority support.`;
  } else {
    explanation += `The predicted class is ${predictedClass}, with a vote share of ${confidence}.`;
  }
  
  return explanation;
}
