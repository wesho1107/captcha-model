"use client";

import * as React from "react";
import { ModelSelector } from "@/components/model-selector";
import { ImageUpload } from "@/components/image-upload";
import { ResultsDisplay } from "@/components/results-display";
import { Button } from "@/components/ui/button";
import { ModelType, PredictionResult } from "@/lib/types";

export default function Home() {
  const [selectedModel, setSelectedModel] = React.useState<ModelType | "">("");
  const [uploadedImage, setUploadedImage] = React.useState<File | null>(null);
  const [imagePreview, setImagePreview] = React.useState<string | null>(null);
  const [results, setResults] = React.useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const handleImageSelect = (file: File) => {
    setUploadedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    setResults(null); // Clear previous results
  };

  const handlePredict = async () => {
    if (!selectedModel || !uploadedImage) return;

    setIsLoading(true);
    
    try {
      const formData = new FormData();
      formData.append("model", selectedModel);
      formData.append("image", uploadedImage);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const result = await response.json();
      setResults(result);
    } catch (error) {
      console.error("Prediction error:", error);
      alert(error instanceof Error ? error.message : "Failed to predict CAPTCHA");
    } finally {
      setIsLoading(false);
    }
  };

  const canPredict = selectedModel !== "" && uploadedImage !== null;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            CAPTCHA Prediction
          </h1>
          <p className="text-gray-600">
            Select a model and upload a CAPTCHA image to get predictions
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-6">
            <ModelSelector
              value={selectedModel}
              onValueChange={(value) => {
                setSelectedModel(value);
                setResults(null);
              }}
            />
            <ImageUpload
              onImageSelect={handleImageSelect}
              imagePreview={imagePreview}
            />
            <Button
              onClick={handlePredict}
              disabled={!canPredict || isLoading}
              className="w-full"
              size="lg"
            >
              {isLoading ? "Predicting..." : "Predict CAPTCHA"}
            </Button>
          </div>

          <div>
            {results && imagePreview && (
              <ResultsDisplay result={results} imageUrl={imagePreview} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
