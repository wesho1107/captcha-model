export interface PredictionResult {
  prediction: string;
  boundingBoxes: Array<{ x1: number; y1: number; x2: number; y2: number }>;
  scores: number[];
  characters: string[];
}

export type ModelType = "CNN" | "ResNet" | "SqueezeNet" | "RCNN";

