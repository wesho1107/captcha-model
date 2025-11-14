import { PredictionResult } from "./types";

// Mock data for testing Phase 1
export const mockPredictions: Record<string, PredictionResult> = {
  CNN: {
    prediction: "abc123",
    boundingBoxes: [
      { x1: 10, y1: 5, x2: 35, y2: 45 },
      { x1: 40, y1: 5, x2: 65, y2: 45 },
      { x1: 70, y1: 5, x2: 95, y2: 45 },
      { x1: 100, y1: 5, x2: 125, y2: 45 },
      { x1: 130, y1: 5, x2: 155, y2: 45 },
      { x1: 160, y1: 5, x2: 185, y2: 45 },
    ],
    scores: [0.95, 0.92, 0.88, 0.91, 0.89, 0.93],
    characters: ["a", "b", "c", "1", "2", "3"],
  },
  ResNet: {
    prediction: "xyz789",
    boundingBoxes: [
      { x1: 12, y1: 8, x2: 38, y2: 48 },
      { x1: 42, y1: 8, x2: 68, y2: 48 },
      { x1: 72, y1: 8, x2: 98, y2: 48 },
      { x1: 102, y1: 8, x2: 128, y2: 48 },
      { x1: 132, y1: 8, x2: 158, y2: 48 },
      { x1: 162, y1: 8, x2: 188, y2: 48 },
    ],
    scores: [0.94, 0.91, 0.87, 0.90, 0.88, 0.92],
    characters: ["x", "y", "z", "7", "8", "9"],
  },
  SqueezeNet: {
    prediction: "test42",
    boundingBoxes: [
      { x1: 15, y1: 10, x2: 40, y2: 50 },
      { x1: 45, y1: 10, x2: 70, y2: 50 },
      { x1: 75, y1: 10, x2: 100, y2: 50 },
      { x1: 105, y1: 10, x2: 130, y2: 50 },
      { x1: 135, y1: 10, x2: 160, y2: 50 },
      { x1: 165, y1: 10, x2: 190, y2: 50 },
    ],
    scores: [0.96, 0.93, 0.89, 0.92, 0.90, 0.94],
    characters: ["t", "e", "s", "t", "4", "2"],
  },
  RCNN: {
    prediction: "captcha",
    boundingBoxes: [
      { x1: 8, y1: 3, x2: 33, y2: 43 },
      { x1: 38, y1: 3, x2: 63, y2: 43 },
      { x1: 68, y1: 3, x2: 93, y2: 43 },
      { x1: 98, y1: 3, x2: 123, y2: 43 },
      { x1: 128, y1: 3, x2: 153, y2: 43 },
      { x1: 158, y1: 3, x2: 183, y2: 43 },
      { x1: 188, y1: 3, x2: 213, y2: 43 },
    ],
    scores: [0.97, 0.94, 0.90, 0.93, 0.91, 0.95, 0.92],
    characters: ["c", "a", "p", "t", "c", "h", "a"],
  },
};

