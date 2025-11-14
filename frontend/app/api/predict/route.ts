import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { writeFile, unlink } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const model = formData.get("model") as string;
    const imageFile = formData.get("image") as File;

    if (!model || !imageFile) {
      return NextResponse.json(
        { error: "Missing model or image" },
        { status: 400 }
      );
    }

    // Validate model name
    const validModels = ["CNN", "ResNet", "SqueezeNet", "RCNN"];
    if (!validModels.includes(model)) {
      return NextResponse.json(
        { error: `Invalid model. Must be one of: ${validModels.join(", ")}` },
        { status: 400 }
      );
    }

    // Save uploaded file temporarily
    const bytes = await imageFile.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const tempFileName = `captcha_${Date.now()}_${Math.random().toString(36).substring(7)}.${imageFile.name.split('.').pop()}`;
    const tempFilePath = join(tmpdir(), tempFileName);

    await writeFile(tempFilePath, buffer);

    try {
      // Determine which Python script to run
      let scriptPath: string;
      let args: string[];

      if (model === "RCNN") {
        scriptPath = join(process.cwd(), "..", "backend", "predict_rcnn.py");
        args = [scriptPath, tempFilePath];
      } else {
        scriptPath = join(process.cwd(), "..", "backend", "predict_segment.py");
        const modelNameMap: Record<string, string> = {
          CNN: "cnn",
          ResNet: "resnet",
          SqueezeNet: "squeezenet",
        };
        args = [scriptPath, modelNameMap[model], tempFilePath];
      }

      // Determine Python executable path (use virtual environment if available)
      const projectRoot = join(process.cwd(), "..");
      const venvPython = process.platform === "win32" 
        ? join(projectRoot, ".venv", "Scripts", "python.exe")
        : join(projectRoot, ".venv", "bin", "python");
      
      // Check if virtual environment exists, otherwise fall back to system Python
      const pythonExecutable = existsSync(venvPython) ? venvPython : "python";

      // Spawn Python process
      const pythonProcess = spawn(pythonExecutable, args, {
        cwd: projectRoot,
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      pythonProcess.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      // Wait for process to complete with timeout
      const timeout = 30000; // 30 seconds
      const result = await Promise.race([
        new Promise<{ code: number | null }>((resolve) => {
          pythonProcess.on("close", (code) => {
            resolve({ code });
          });
        }),
        new Promise<{ code: number | null }>((_, reject) => {
          setTimeout(() => {
            pythonProcess.kill();
            reject(new Error("Python script execution timeout"));
          }, timeout);
        }),
      ]);

      // Clean up temp file
      await unlink(tempFilePath).catch(() => {
        // Ignore errors when deleting temp file
      });

      if (result.code !== 0) {
        return NextResponse.json(
          { error: `Python script failed: ${stderr || stdout}` },
          { status: 500 }
        );
      }

      // Parse JSON output
      try {
        const predictionResult = JSON.parse(stdout.trim());
        
        // Check if there's an error in the result
        if (predictionResult.error) {
          return NextResponse.json(
            { error: predictionResult.error },
            { status: 500 }
          );
        }

        return NextResponse.json(predictionResult);
      } catch (parseError) {
        return NextResponse.json(
          { error: `Failed to parse Python output: ${stdout}` },
          { status: 500 }
        );
      }
    } catch (error) {
      // Clean up temp file on error
      await unlink(tempFilePath).catch(() => {
        // Ignore errors when deleting temp file
      });

      if (error instanceof Error) {
        return NextResponse.json(
          { error: error.message },
          { status: 500 }
        );
      }
      return NextResponse.json(
        { error: "Unknown error occurred" },
        { status: 500 }
      );
    }
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

