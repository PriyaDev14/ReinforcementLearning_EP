/*
  Optional ESP32 prototype.
  Reads temperature + occupancy and demonstrates the hardware interface.
  The Python RL model remains on the local computer/server.

  Planned hardware:
  ESP32 + DHT22 + PIR + relay/smart plug + controlled appliance.
*/

#include <WiFi.h>
#include "DHT.h"

#define DHTPIN 4
#define DHTTYPE DHT22
#define PIR_PIN 27
#define RELAY_PIN 26

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  dht.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void loop() {
  float temperature = dht.readTemperature();
  int occupancy = digitalRead(PIR_PIN);

  if (isnan(temperature)) {
    Serial.println("Temperature read failed");
    delay(2000);
    return;
  }

  Serial.printf("Temperature: %.2f C | Occupancy: %d\n",
                temperature, occupancy);

  // In a full deployment:
  // 1. Send sensor values to the local HVAC RL controller.
  // 2. Train locally.
  // 3. Send only model parameters to the FL server.
  // 4. Receive aggregated parameters.
  // 5. Apply the selected appliance command.

  delay(5000);
}
