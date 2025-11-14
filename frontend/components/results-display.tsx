"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PredictionResult } from "@/lib/types";

interface ResultsDisplayProps {
  result: PredictionResult | null;
  imageUrl: string | null;
}

export function ResultsDisplay({ result, imageUrl }: ResultsDisplayProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    if (!result || !imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes and labels
      result.boundingBoxes.forEach((box, index) => {
        const { x1, y1, x2, y2 } = box;
        const char = result.characters[index] || "";
        const score = result.scores[index] || 0;

        // Draw bounding box
        ctx.strokeStyle = "#ef4444"; // red-500
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw label background
        const labelText = `${char}: ${(score).toFixed(1)}`;
        ctx.font = "bold8px sans-serif";
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;
        const textHeight = 8; 

        ctx.fillStyle = "rgba(239, 68, 68, 0.0)"; // red-500 with opacity
        ctx.fillRect(
          x1,
          y1 - textHeight,
          textWidth,
          textHeight
        );

        // Draw label text
        ctx.fillStyle = "red"; // red-500 with opacity
        ctx.fillText(labelText, x1, y1 - 2);
      });
    };
    img.src = imageUrl;
  }, [result, imageUrl]);

  if (!result) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <p className="text-sm text-gray-500 mb-2">Predicted CAPTCHA:</p>
          <p className="text-3xl font-bold text-primary">{result.prediction}</p>
        </div>
        {imageUrl && (
          <div className="flex justify-center">
            <canvas
              ref={canvasRef}
              className="max-w-full h-auto border rounded"
              style={{ maxHeight: "500px" }}
            />
          </div>
        )}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="font-semibold">Confidence Scores:</p>
            <div className="flex flex-wrap gap-2 mt-2">
              {result.scores.map((score, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-gray-100 rounded"
                >
                  {result.characters[index]}: {(score * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="font-semibold">Average Confidence:</p>
            <p className="text-lg font-bold text-primary mt-2">
              {((result.scores.reduce((a, b) => a + b, 0) / result.scores.length) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

