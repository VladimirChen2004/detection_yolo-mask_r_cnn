"""
Модуль для детекции людей на видео с использованием YOLOv11 и Mask R-CNN.

Этот модуль предоставляет функционал для обработки видео с детекцией людей
с помощью двух современных архитектур: YOLOv11 (one-stage) и Mask R-CNN (two-stage).
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from typing import Dict, Tuple, Optional
import argparse
from pathlib import Path


class YOLODetector:
    """
    Класс для детекции людей с использованием YOLOv11.
    
    Attributes:
        model: Предобученная модель YOLOv11
        conf_threshold (float): Порог уверенности для детекции
        iou_threshold (float): Порог IoU для NMS
    """
    
    def __init__(
        self, 
        model_path: str = 'yolo11m.pt',
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45
    ):
        """
        Инициализация детектора YOLOv11.
        
        Args:
            model_path (str): Путь к весам модели
            conf_threshold (float): Порог уверенности (0.0-1.0)
            iou_threshold (float): Порог IoU для NMS (0.0-1.0)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Детектирует людей на кадре.
        
        Args:
            frame (np.ndarray): Входной кадр в формате BGR
            
        Returns:
            Tuple[np.ndarray, int]: Аннотированный кадр и количество людей
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Класс 'person' в COCO
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        num_persons = len(results[0].boxes)
        
        return annotated_frame, num_persons
    
    def process_video(
        self,
        video_path: str,
        output_path: str
    ) -> Dict[str, float]:
        """
        Обрабатывает видео с детекцией людей.
        
        Args:
            video_path (str): Путь к входному видео
            output_path (str): Путь для сохранения результата
            
        Returns:
            Dict[str, float]: Статистика обработки (avg, min, max людей)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Настраиваем VideoWriter для сохранения результата
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_persons = []
        
        # Обрабатываем видео покадрово
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция людей на кадре
            annotated_frame, num_persons = self.detect(frame)
            total_persons.append(num_persons)
            
            # Добавляем текстовую информацию на кадр
            cv2.putText(
                annotated_frame,
                f'Людей: {num_persons}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            out.write(annotated_frame)
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        
        # Возвращаем статистику
        return {
            'avg_persons': np.mean(total_persons),
            'max_persons': np.max(total_persons),
            'min_persons': np.min(total_persons)
        }


class MaskRCNNDetector:
    """
    Класс для детекции людей с использованием Mask R-CNN.
    
    Attributes:
        model: Предобученная модель Mask R-CNN
        device: Устройство для вычислений (cuda/cpu)
        score_threshold (float): Порог уверенности для детекции
    """
    
    def __init__(
        self,
        score_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Инициализация детектора Mask R-CNN.
        
        Args:
            score_threshold (float): Порог уверенности (0.0-1.0)
            device (str, optional): Устройство ('cuda' или 'cpu')
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        self.score_threshold = score_threshold
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Детектирует людей на кадре.
        
        Args:
            frame (np.ndarray): Входной кадр в формате BGR
            
        Returns:
            Dict: Словарь с предсказаниями (boxes, labels, scores)
        """
        # Преобразуем BGR в RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Нормализуем и преобразуем в тензор
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image_rgb).to(self.device)
        
        # Детекция
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        return predictions
    
    def draw_boxes(
        self,
        frame: np.ndarray,
        predictions: Dict
    ) -> Tuple[np.ndarray, int]:
        """
        Отрисовывает боксы на кадре.
        
        Args:
            frame (np.ndarray): Входной кадр
            predictions (Dict): Предсказания модели
            
        Returns:
            Tuple[np.ndarray, int]: Аннотированный кадр и количество людей
        """
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        num_persons = 0
        
        # Отрисовываем только людей (label=1 в COCO) выше порога
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > self.score_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                # Рисуем прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с классом и уверенностью
                text = f'person {score:.2f}'
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                num_persons += 1
        
        return frame, num_persons
    
    def process_video(
        self,
        video_path: str,
        output_path: str
    ) -> Dict[str, float]:
        """
        Обрабатывает видео с детекцией людей.
        
        Args:
            video_path (str): Путь к входному видео
            output_path (str): Путь для сохранения результата
            
        Returns:
            Dict[str, float]: Статистика обработки
        """
        cap = cv2.VideoCapture(video_path)
        
        # Получаем параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Настраиваем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_persons = []
        
        # Обрабатываем видео покадрово
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция
            predictions = self.detect(frame)
            
            # Отрисовка боксов
            annotated_frame, num_persons = self.draw_boxes(
                frame.copy(),
                predictions
            )
            total_persons.append(num_persons)
            
            # Сохраняем кадр
            out.write(annotated_frame)
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        
        return {
            'avg_persons': np.mean(total_persons),
            'max_persons': np.max(total_persons),
            'min_persons': np.min(total_persons)
        }


def main():
    """
    Главная функция - точка входа в программу.
    
    Обрабатывает аргументы командной строки и запускает детекцию.
    """
    parser = argparse.ArgumentParser(
        description='Детекция людей на видео с помощью YOLOv11 или Mask R-CNN'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo', 'maskrcnn'],
        required=True,
        help='Выбор модели: yolo или maskrcnn'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Путь к входному видео'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Путь для сохранения результата'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Порог уверенности (для YOLO: 0.3, для Mask R-CNN: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='Порог IoU для NMS (только для YOLO)'
    )
    
    args = parser.parse_args()
    
    # Проверяем существование входного файла
    if not Path(args.input).exists():
        print(f"Ошибка: файл {args.input} не найден")
        return
    
    print(f"Обработка видео с моделью {args.model.upper()}...")
    print(f"Входной файл: {args.input}")
    print(f"Выходной файл: {args.output}")
    
    # Запуск соответствующей модели
    if args.model == 'yolo':
        detector = YOLODetector(
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        stats = detector.process_video(args.input, args.output)
        
    elif args.model == 'maskrcnn':
        detector = MaskRCNNDetector(score_threshold=args.conf)
        stats = detector.process_video(args.input, args.output)
    
    # Выводим статистику
    print("\n=== Статистика обработки ===")
    print(f"Среднее количество людей: {stats['avg_persons']:.2f}")
    print(f"Максимум людей на кадре: {stats['max_persons']:.0f}")
    print(f"Минимум людей на кадре: {stats['min_persons']:.0f}")
    print(f"\nРезультат сохранен в: {args.output}")


if __name__ == "__main__":
    main()
