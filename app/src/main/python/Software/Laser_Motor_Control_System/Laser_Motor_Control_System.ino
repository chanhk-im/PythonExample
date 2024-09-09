#include <Servo.h>

Servo myservo;  // 서보 객체 생성
int pos = 90;    // 서보의 현재 위치

void setup() {
  myservo.attach(A1);  // 서보 모터를 아두이노의 9번 핀에 연결
  pinMode(A0, OUTPUT);  // A0 핀을 출력으로 설정
  digitalWrite(A0, HIGH);  // A0 핀을 HIGH로 설정
}

void loop() {
  for (pos = 0; pos <= 40; pos += 1) {
    myservo.write(pos);  // 서보 위치 설정
    delay(400);         // 1초 대기
  }

  delay(400);  // 1초 대기

  // 오른쪽에서 왼쪽으로 90도 이동
  for (pos = 40; pos >= 0; pos -= 1) {
    myservo.write(pos);  // 서보 위치 설정
    delay(400);         // 1초 대기
  }

  delay(400);  // 1초 대기
}
