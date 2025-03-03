import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';
import 'package:image/image.dart' as img;

void main() {
  runApp(const BlurDetectionApp());
}

class BlurDetectionApp extends StatefulWidget {
  const BlurDetectionApp({Key? key}) : super(key: key);

  @override
  _BlurDetectionAppState createState() => _BlurDetectionAppState();
}

class _BlurDetectionAppState extends State<BlurDetectionApp> {
  File? _image;
  Interpreter? _interpreter;
  String _result = "Seleziona un'immagine per l'analisi";

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/blur_model_with_metrics.tflite');
      setState(() {});
    } catch (e) {
      setState(() {
        _result = "Errore nel caricamento del modello: $e";
      });
    }
  }

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null) {
      setState(() {
        _image = File(result.files.single.path!);
      });
      _analyzeImage();
    }
  }

  Future<void> _analyzeImage() async {
    if (_image == null || _interpreter == null) return;

    Uint8List imageBytes = await _image!.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) return;

    final int inputSize = 224;
    img.Image resizedImage = img.copyResize(image, width: inputSize, height: inputSize);

    // 📌 Convertiamo l'immagine in un tensore 4D [1, 224, 224, 3]
    List<List<List<List<double>>>> input = List.generate(
      1,
          (batch) => List.generate(
        inputSize,
            (y) => List.generate(
          inputSize,
              (x) {
            final pixel = resizedImage.getPixel(x, y);
            return [
              pixel.r / 255.0,
              pixel.g / 255.0,
              pixel.b / 255.0
            ];
          },
        ),
      ),
    );

    // 📊 Calcola metriche (Laplacian, PSNR, SSIM) - **Migliorare con OpenCV**
    double laplacianVar = 100.0; // Valore placeholder, da migliorare con OpenCV su PC
    double psnrValue = 30.0; // Placeholder
    double ssimValue = 0.8; // Placeholder

    List<List<double>> metrics = [[laplacianVar, psnrValue, ssimValue]];
    List<List<double>> output = [[0.0]];

    // Esegui l'inferenza con il modello
    _interpreter!.run([input, metrics], output);

    setState(() {
      _result = "Sfocatura: ${output[0][0] > 0.5 ? 'Sfocata' : 'Nitida'}";
    });
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text("Blur Detection su PC")),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image != null ? Image.file(_image!) : const Icon(Icons.image, size: 100),
            const SizedBox(height: 20),
            Text(_result, style: const TextStyle(fontSize: 18)),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text("Seleziona Immagine"),
            ),
          ],
        ),
      ),
    );
  }
}
