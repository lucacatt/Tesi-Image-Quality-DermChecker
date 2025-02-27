import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
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
  String _result = "Carica un'immagine per l'analisi";

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      setState(() {});
    } catch (e) {
      setState(() {
        _result = "Errore nel caricamento del modello: $e";
      });
    }
  }

  Future<void> _pickImage() async {
    final pickedFile =
        await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      _analyzeImage();
    }
  }

  Future<void> _analyzeImage() async {
    if (_image == null || _interpreter == null) return;

    // Leggi il file immagine
    Uint8List imageBytes = await _image!.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);

    if (image == null) {
      setState(() {
        _result = "Errore nella decodifica dell'immagine";
      });
      return;
    }

    // Normalizza e ridimensiona l'immagine secondo il modello
    final int inputSize = 224; // Verifica la dimensione richiesta dal modello
    img.Image resizedImage = img.copyResize(image, width: inputSize, height: inputSize);

    // Converti l'immagine in un tensore 4D [1, 224, 224, 3]
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

    // **Modifica l'output per supportare la dimensione `[1,2]`**
    List<List<double>> output = List.generate(1, (_) => List.filled(2, 0.0));

    // Esegui l'inferenza
    _interpreter!.run(input, output);

    setState(() {
      double blurScore = output[0][0];  // Adatta se il modello restituisce altro
      double sharpScore = output[0][1];

      _result = "Sfocata: ${blurScore.toStringAsFixed(4)}, Nitida: ${sharpScore.toStringAsFixed(4)}";
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
        appBar: AppBar(title: const Text("Blur Detection con TFLite")),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image != null
                ? Image.file(_image!)
                : const Icon(Icons.image, size: 100),
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
