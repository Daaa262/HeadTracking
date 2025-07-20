import cv2

# Inicjalizacja przechwytywania wideo (0 - domyślna kamerka)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nie udało się otworzyć kamerki.")
    exit()

while True:
    # Przechwycenie klatki
    ret, frame = cap.read()

    if not ret:
        print("Nie udało się odczytać klatki.")
        break

    # Wyświetlenie obrazu w oknie
    cv2.imshow("Podgląd z kamerki", frame)

    # Przerwanie pętli po wciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
