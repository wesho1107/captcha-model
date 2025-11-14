"use client";

import * as React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { ModelType } from "@/lib/types";

interface ModelSelectorProps {
  value: ModelType | "";
  onValueChange: (value: ModelType) => void;
}

export function ModelSelector({ value, onValueChange }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="model-select">Select Model</Label>
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger id="model-select" className="w-full">
          <SelectValue placeholder="Choose a model" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="CNN">CNN</SelectItem>
          <SelectItem value="ResNet">ResNet</SelectItem>
          <SelectItem value="SqueezeNet">SqueezeNet</SelectItem>
          <SelectItem value="RCNN">RCNN</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

