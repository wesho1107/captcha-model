"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  imagePreview: string | null;
}

export function ImageUpload({ onImageSelect, imagePreview }: ImageUploadProps) {
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = React.useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.type === "image/png" || file.type === "image/jpeg")) {
      onImageSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type === "image/png" || file.type === "image/jpeg")) {
      onImageSelect(file);
    }
  };

  return (
    <div className="space-y-2">
      <Label htmlFor="image-upload">Upload CAPTCHA Image</Label>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? "border-primary bg-primary/5"
            : "border-gray-300 hover:border-gray-400"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {imagePreview ? (
          <div className="space-y-4">
            <img
              src={imagePreview}
              alt="Preview"
              className="max-w-full max-h-64 mx-auto rounded"
            />
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
            >
              Change Image
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-gray-500">
              Drag and drop an image here, or click to select
            </p>
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
            >
              Select Image
            </Button>
          </div>
        )}
        <Input
          ref={fileInputRef}
          id="image-upload"
          type="file"
          accept="image/png,image/jpeg"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
      <p className="text-sm text-gray-500">
        Supported formats: PNG, JPG (Max size: 10MB)
      </p>
    </div>
  );
}

