import argparse
import os
import sys
from collections import namedtuple
from enum import Enum

import cv2
import pandas as pd
from ultralytics import YOLO

Capture = namedtuple('Capture', ['x1', 'y1', 'x2', 'y2'])
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


class TableStatus(Enum):
    BUSY = (0, 255, 0), "Table is busy", 0 
    FREE = (255, 0, 0), "Table is free", 1
    NEAR = (0, 0, 255), "Peson near the table", 2


class Error:
    USAGE_ERROR = ("Использование: main --video [путь к файлу] "
                   "--report [тип обработки данных]")
    FILE_NOT_EXIST_ERROR = "Проверьте правильность указания имени файла"


def exit_with_error(msg: str) -> None:
    """
        Печатает сообщение в терминал и завершает процесс
        
        msg: str - сообщение для печати
    """
    print(msg)
    sys.exit(1)


def person_table_intersection(table: Capture,
                              persons: list[Capture]) -> bool:
    """
        Получает данные bbox определенных нейросетью для стола.
        Определяет есть ли геометрическое пересечение bbox.

        table: Capture - координаты стола
        persons: list[Capture] - координаты персон
    """

    if table is None or not persons:
        return False
    
    for person in persons:
        if (min(table.x2, person.x2) > max(table.x1, person.x1) and 
            min(table.y2, person.y2) > max(table.y1, person.y1)):
            return True
    
    return False


def main():
    """
        Получает данные из коммандной строки:
        --video - список имен файлов
        --output - путь, куда сохранить видео
        
        Получает данные от пользователя.
        Обрабатывет видео:
            - останавливается для выбора пользователем фрейма
            - с помощью нейросети определяет два класса во фрейме: 
            персона и стол. Единожды определив координаты стола, больше их не 
            меняем, чтобы изображение было более стабильным. 
            Тем более, что объект статичный
            - используем координаты bbox от нейросети для формирования 
            статуса столика и рисуем соответсвующнго цвета bbox
                -- стол занят - есть персона и стол, координаты пересекаются
                (Зеленый)
                -- человек около стола - координаты не пересекаюся
                (Красный)
                -- стол пуст - персона отсутсвует или не определена
                (Синий)
            - пишем обработанные фреймы в файл
            - записываем временные метки для каждого статуса и далее формируем 
            датафрейм
            - анализируем данные измений статуса для вычисления простоя столика
    """
    # Получаем данные от пользователя
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', metavar='VIDEO', type=str,
                        help='Input video')
    parser.add_argument('--output', default='output.mp4', metavar='OUTPUT', 
                        type=str, help='Output path')

    args = parser.parse_args()
    if not args.video: 
        exit_with_error(Error.USAGE_ERROR)

    if not os.path.exists(args.video):
        exit_with_error(Error.FILE_NOT_EXIST_ERROR)
    
    # Путь для выходного файла по умолчанию или аргументов

    # Инициалищация модели, фрейма и выходного файла
    model = YOLO(os.path.join(PROJECT_DIR, "yolov8n.pt"))
    cap = cv2.VideoCapture(args.video)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output, fourcc, 20, 
                          (frame_width, frame_height))
    
    table_status = {"time": [], "status": []}

    ret, frame = cap.read()
    if ret:
        # Выбор сегмента
        bbox = cv2.selectROI("select frame", frame, fromCenter=False)
        cv2.destroyWindow("select frame") 

        x, y, w, h = bbox
        founded_table = None
        color = 255, 0, 0
        if w > 0 and h > 0:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame = frame[y:y + h, x:x + w]
                # 0 - persons, 60 - dining table
                results = model.predict(cropped_frame, classes=[0, 60])

                # Находим координаты bbox для каждого класса и пакуем в словарь
                # Словарь имя класса: список координат. 
                # Иногда может быть несколько персон
                coords: dict[str, list[Capture]] = {}
                for i in range(len(results[0].boxes.cls.int())):
                    coords.setdefault(
                        results[0].names[results[0].boxes.cls.int()[i].item()], 
                        []
                    ).append(
                        [int(x) for x in results[0].boxes.xyxy[i].tolist()])
                
                text = None
                persons = []
                # Если мы еще не определили координаты стола, и они есть в 
                # полученных для фрейма, то определяем 
                if not founded_table and coords.get("dining table"):
                    founded_table = Capture._make(
                        coords.get("dining table")[0])
                elif founded_table:
                    # координаты для стола есть - можно определять статус     
                    if coords.get("person"):
                        persons = [Capture._make(person) 
                                   for person in coords.get("person")]
                    if not coords:
                        color, text, idx = TableStatus.FREE.value
                    elif person_table_intersection(founded_table, persons):
                        color, text, idx = TableStatus.BUSY.value
                    elif (persons and 
                          not person_table_intersection(founded_table, persons)
                          ):
                        color, text, idx = TableStatus.NEAR.value
                        
                    cv2.rectangle(cropped_frame,  
                                  (founded_table.x1, founded_table.y1), 
                                  (founded_table.x2, founded_table.y2), 
                                  color, 2)
                    cv2.putText(cropped_frame, text, 
                                (founded_table.x1, founded_table.y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    # Записывем статус и метки в секундах
                    table_status["time"].append(
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                    table_status["status"].append(idx)
                out.write(frame)
                cv2.imshow("Frame with bbox", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("cancel")
    else:
        print("Ошибка чтения видео")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    """
    Обработка результатов
    Статусы:
        0 - столик занят
        1 - столик свободен
        2 - человек около стола
    """
    df = pd.DataFrame(table_status)
    # Так как нужно определить "среднее время между уходом гостя и подходом
    # следующего человека", то можно приравнять 'Столик занят' и
    # 'человек около'. Здесь по разному можно трактовать;
    # мы не сможем определить какой человек у стола: тот же или другой
    # посетитель. В любом случае статус 'человек около стола' будет лишним.
    df['status'] = df['status'].replace(2, 0)
    # Находим ряды с изменением статуса
    df = df[df.status != df.status.shift()]
    # Находим разницу между временными метками измений статуса
    df['difference'] = df['time'].diff()
    # Так как статус привязан к работе нейросети, то получается большое
    # количество шума(ложных срабатываний).
    # Уменьшаем количество шума, выбирая ряды с разницей больше секунды
    df = df[df['difference'] > 1]
    # Ряды со статусом 0 - это событие, когда столик стал занят после того
    # как он был свободен. И разница указывает на количество времени
    # простаивания. Вычисляем среднее время простаивания
    print(f'Среднее время простоя столика: \
          {df[df['status'] == 0]['difference'].mean()} секунд')
       

if __name__ == "__main__":
    main()
