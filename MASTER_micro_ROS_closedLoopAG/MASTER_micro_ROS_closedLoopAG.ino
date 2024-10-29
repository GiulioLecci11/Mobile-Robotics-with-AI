/*Motor controller using micro_ros serial set_microros_transports*/
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/int32.h>
#include <odometry.h>
#include <nav_msgs/msg/odometry.h>
#include <geometry_msgs/msg/twist.h>
#include <geometry_msgs/msg/vector3.h>
#include <ESP32Encoder.h>
#include <math.h>
// LED USATI PER DEBUG
#define BLUE 4
#define ORANGE 16
#define WHITE 17
#define YELLOW 5

#define LED_LABORATORIO 17
// MAPPATURA PIN DELL'ESP32 SUI DUE DRIVE DEI MOTORI
#define LEFT_MOTOR 33
#define RIGHT_MOTOR 32
#define LEFT_DIRECTION 13
#define RIGHT_DIRECTION 26  // 17, pin che abbiamo fatto saldare da Ciccio ma non funziona (forse bruciato) e in generale abbiamo capito che i motori sono montati "al contrario" quindi i pin direction sono giusti

#define WHEEL_BASE .40  // 40cm nel nostro robot

#define ENCODER_SINGLE_ROTATION 100000  // da codice ciccio

#define LEFT_ENCODER_A 12
#define RIGHT_ENCODER_A 14
#define LEFT_ENCODER_B 27   // Da capire cosa è pin A e B rispetto a 1 e 2 dell'altro file
#define RIGHT_ENCODER_B 25  // No scherzo non ci serve perché qui l'encoder viene gestito dalla classe ESP32Encoder

#define LEFT_PWM_CHANNEL 0
#define RIGHT_PWM_CHANNEL 1  // 2 Pin coi quali mandiamo i comandi ai motori
#define MOTOR_PWM_FREQ 20000
#define MOTOR_PWM_RES 8  // Risoluzione PWM bit

// parameters of the robot
float wheels_y_distance_ = WHEEL_BASE;
float wheel_radius = 0.1;
float wheel_circumference_ = 2 * M_PI * wheel_radius;

// encoder value per revolution of left wheel and right wheel
int tickPerRevolution_LW = 1055;  // NON SONO INUTILI?????????????????????????????????? si perché usiamo la macro definita sopra
int tickPerRevolution_RW = 1055;
int threshold = 16;  // Trovata manualmente tramite programma di ciccio (metti valore di PWM e vedi se ruote girano, per valori <=16 non gira)


rcl_subscription_t subscriber;  // subscriber dei messaggi cmd_vel
geometry_msgs__msg__Twist msg;
rclc_executor_t executor;
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;
rcl_publisher_t odom_publisher;
std_msgs__msg__Int32 encodervalue_l;  // NON SONO INUTILI??????????????? Si, era l'implementazione del tizio del tutorial online
std_msgs__msg__Int32 encodervalue_r;
nav_msgs__msg__Odometry odom_msg;
rcl_timer_t timer;
rcl_timer_t ControlTimer;
unsigned long long time_offset = 0;
unsigned long prev_cmd_time = 0;
unsigned long prev_odom_update = 0;
Odometry odometry;

// Class to estimate velocity of encoder
class VelEstimator {
public:
  VelEstimator(ESP32Encoder *encoder)
    : encoder_(encoder) {
    lastTime = millis();
    encoder->clearCount();
  }

  float estimate_velocity() {
    long position = encoder_->getCount();
    encoder_->clearCount();
    int time = millis();
    float velocity = position / (time - lastTime) * 1000 / ENCODER_SINGLE_ROTATION * 2 * PI;  // measures in rad/s. Il ragionamento sarebbe # (ticks) / (millisencondi) * (millisencondi in un secondo) / (ticks/rivoluzione) * (radianti/rivoluzione)
    lastTime = time;

    return velocity;
  }

private:
  ESP32Encoder *encoder_;
  int lastTime;
};

// creating a class for motor control
class MotorController {
public:
  int8_t motorPin;
  int8_t motorChannel;  // PER PWM
  int8_t direction;
  VelEstimator *vel_estimator;
  volatile long CurrentPosition = 0;
  volatile long PreviousPosition = 0;  // CURRENT E PREVIOUS POS INUTILI????????????????? Si, non ricordo da cosa sono stati tolti e perché
  volatile long CurrentTime = 0;
  volatile long PreviousTime = 0;
  volatile long CurrentTimeforError = 0;
  volatile long PreviousTimeForError = 0;
  float rpmFilt = 0.0;  // =0.0 aggiunto da me
  float eintegral = 0.0;
  float ederivative = 0.0;
  float rpmPrev = 0.0;
  float kp;
  float ki;
  float kd;  // Questi sono attributi della classe MotorController (ne istanzieremo 2 entità), poi gli passeremo i valori dichiarati sopra
  float error = 0.0;
  float previousError = 0.0;
  int tick;
  ESP32Encoder *encoder;

  MotorController(int8_t pwmChannel, int8_t DirectionPin, int8_t encoder_pinA, int8_t encoder_pinB, int8_t motor_pin) {  // costruttore dove inizializzo la maggior parte dei parametri
    this->motorPin = motor_pin;
    this->encoder = new ESP32Encoder();
    this->encoder->attachFullQuad(encoder_pinA, encoder_pinB);
    this->vel_estimator = new VelEstimator(encoder);
    this->motorChannel = pwmChannel;
    this->direction = DirectionPin;
    pinMode(DirectionPin, OUTPUT);
    digitalWrite(direction, HIGH);
    // initializing pwm signal parameters
    ledcSetup(pwmChannel, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcAttachPin(motor_pin, pwmChannel);  // metterei invece ldcAttachPin(LEFT_MOTOR, LEFT_PWM_CHANNEL); -(giulio) Posso anche essere d'accordo, e quindi differenziare tra i due motori con l'if motor pin, ma perché???????
    this->tick = ENCODER_SINGLE_ROTATION;  // Dovrebbe essere lui che vuole, prima gli passava il tick per revolution quindi...
    ledcWrite(pwmChannel, 0);
    // Serial.println("fine costruttore");
  }

  // initializing the parameters of PID controller
  void initPID(float proportionalGain, float integralGain, float derivativeGain) {
    kp = proportionalGain;
    ki = integralGain;
    kd = derivativeGain;
  }

  // function return rpm of the motor using the encoder tick values
  float getRpm() {  // tic sarebbe tic per revolution

    float velocity = vel_estimator->estimate_velocity();
    float rpm = (velocity / (2 * PI)) * 60;                        // sarebbe (rad/s) / (2*PI rad/giro) * (60s/min)   = giri/min
    //rpmFilt = 0.854 * rpmFilt + 0.0728 * rpm + 0.0728 * rpmPrev;   // Filtro passa basso
    //rpmFilt = 0.3594 * rpmFilt + 0.3203 * rpm + 0.3203 * rpmPrev;  // Nostro (Fc = 15Hz)
    rpmFilt=(4776463164401*rpm)/35184372088832 + (6561650114567681*rpmFilt)/9007199254740992 + (1222774570086655*rpmPrev)/9007199254740992;
    //rpmFilt=rpm; // Senza filtro
    rpmPrev = rpm;
    PreviousTime = CurrentTime;
    // Serial.println(rpmFilt);
    return rpmFilt;
  }

  // pid controller
  float pid(float setpoint, float feedback) {
    CurrentTimeforError = millis();
    float delta2 = ((float)CurrentTimeforError - PreviousTimeForError) / 1.0e3;
    error = setpoint - feedback;
    eintegral = eintegral + (error * delta2);
    ederivative = (error - previousError) / delta2;
    float control_signal = (kp * error) + (ki * eintegral) + (kd * ederivative);

    previousError = error;
    PreviousTimeForError = CurrentTimeforError;
    return control_signal;
  }

  // move the robot wheels based the control signal generated by the pid controller
  void moveBase(float ActuatingSignal, int threshold) {
    //(RIGHT HIGH, LEFT LOW) -> Forward (motori montati "al contrario" come detto prima)
    //(RIGHT LOW, LEFT HIGH) -> Backward
    if (ActuatingSignal > 0)  // Forward
    {
      if (this->motorPin == RIGHT_MOTOR) {
        digitalWrite(direction, HIGH);  // invertita se motor right
      } else
        digitalWrite(direction, LOW);  // normale scrittura sul motor left
    } else                             // Backward
    {
      if (this->motorPin == RIGHT_MOTOR) {
        digitalWrite(direction, LOW);
      } else
        digitalWrite(direction, HIGH);  // potrebbe essere al contrario
    }
    int pwm = threshold + (int)fabs(ActuatingSignal);  // così che anche il minimo segnale di PWM calcolato dal PID faccia muovere le ruote perché sempre sopra threshold (16)
    if (pwm > 255)
      pwm = 255;
    ledcWrite(motorChannel, pwm);
  }
  void stop() {
    digitalWrite(direction, LOW);
    ledcWrite(motorChannel, 0);
  }

  // void plot(float Value1, float Value2){
  //     Serial.print("Value1:");
  //     Serial.print(Value1);
  //     Serial.print(",");
  //     Serial.print("value2:");
  //     Serial.println(Value2);
  // }
};  // fine classe motor controller

// creating objects for right wheel and left wheel
// SONO INUTILI (alberto)
ESP32Encoder *left_encoder;
ESP32Encoder *right_encoder;
VelEstimator *left_vel_estimator;
VelEstimator *right_vel_estimator;

// creating objects for right wheel and left wheel
// MotorController leftWheel(L_FORW, L_BACK, L_enablePin, L_encoderPin1, L_encoderPin2, tickPerRevolution_LW);         ELIMINIAMO MUNNIZZA????? si
// MotorController rightWheel(R_FORW, R_BACK, R_enablePin, R_encoderPin1, R_encoderPin2, tickPerRevolution_RW);

MotorController rightWheel(RIGHT_PWM_CHANNEL, RIGHT_DIRECTION, RIGHT_ENCODER_A, RIGHT_ENCODER_B, RIGHT_MOTOR);
MotorController leftWheel(LEFT_PWM_CHANNEL, LEFT_DIRECTION, LEFT_ENCODER_B, LEFT_ENCODER_A, LEFT_MOTOR);

#define LED_PIN 2
#define RCCHECK(fn) \
  { \
    rcl_ret_t temp_rc = fn; \
    if ((temp_rc != RCL_RET_OK)) { \
      error_loop(); \
    } \
  }
#define RCSOFTCHECK(fn) \
  { \
    rcl_ret_t temp_rc = fn; \
    if ((temp_rc != RCL_RET_OK)) { \
      error_loop(); \
    } \
  }

void error_loop() {
  while (1) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// subscription callback function

void setup() {
  // pid constants of left wheel
  float kp_r = 6.0;
  float ki_r = 0.9;
  float kd_r = 0.05;
  // pid constants of right wheel (same as the left ones)
  /*float kp_r = kp_l;
  float ki_r = ki_l;
  float kd_r = kd_l;*/

  //float kp_l = 7.0;
  // float ki_l = 0.5;
  //float kd_l = 0.25;


  float kp_l = kp_r;
  float ki_l = ki_r;
  float kd_l = kd_r;
  // pinMode(ORANGE, OUTPUT);
  // pinMode(YELLOW, OUTPUT);
  // pinMode(BLUE, OUTPUT);
  // pinMode(WHITE, OUTPUT);
  // pinMode(LED_LABORATORIO, OUTPUT);                                    ELIMINIAMO LA MUNNIZZA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Serial.println("inizio");
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  // Serial.println("Ho finito con encoder");
  /*
  //initializing the pid constants
  leftWheel.initPID(kp_l, ki_l, kd_l);
  rightWheel.initPID(kp_r, ki_r, kd_r);
  //initializing interrupt functions for counting the encoder tick values
  attachInterrupt(digitalPinToInterrupt(leftWheel.EncoderPinB), updateEncoderL, RISING);
  attachInterrupt(digitalPinToInterrupt(rightWheel.EncoderPinA), updateEncoderR, RISING);
  //initializing pwm signal parameters
  ledcSetup(pwmChannelL, freq, resolution);
  ledcAttachPin(leftWheel.Enable, pwmChannelL);
  ledcSetup(pwmChannelR, freq, resolution);
  ledcAttachPin(rightWheel.Enable, pwmChannelR);*/

  //**********************IL NOSTRO CONTROLLERZZZ***************************

  // Serial.println("finito motorcontroller2 left");

  // initializing the pid constants
  leftWheel.initPID(kp_l, ki_l, kd_l);
  rightWheel.initPID(kp_r, ki_r, kd_r);

  //************************************************************************

  set_microros_transports();  // check che ci sia comunicazione tra ros e microros, da qui in poi serve il robot per eseguire
  // pinMode(LED_PIN, OUTPUT);
  // digitalWrite(LED_PIN, HIGH);

  delay(2000);

  allocator = rcl_get_default_allocator();
  // digitalWrite(BLUE, HIGH);
  // delay(500);                                    //ELIMINIAMO TUTTI STI WRITE SUI LED PER DEBUG!!!!!

  // create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  // digitalWrite(BLUE, LOW);

  // create node
  RCCHECK(rclc_node_init_default(&node, "micro_ros_esp32_node", "", &support));

  // create subscriber for cmd_vel topic
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "cmd_vel"));

  // create a odometry publisher
  RCCHECK(rclc_publisher_init_default(
    &odom_publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(nav_msgs, msg, Odometry),
    "odom/unfiltered"));

  // timer function for controlling the motor base. At every samplingT time
  // MotorControll_callback function is called
  // Here we set SamplingT=20 Which means at every 20 milliseconds MotorControll_callback function is called
  const unsigned int samplingT = 20;
  RCCHECK(rclc_timer_init_default(
    &ControlTimer,
    &support,
    RCL_MS_TO_NS(samplingT),
    MotorControll_callback));

  // create executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg, &subscription_callback, ON_NEW_DATA));
  // RCCHECK(rclc_executor_add_timer(&executor, &timer));
  RCCHECK(rclc_executor_add_timer(&executor, &ControlTimer));
}

void loop() {
  // put your main code here, to run repeatedly:
  // digitalWrite(WHITE, !digitalRead(WHITE));

  delay(100);
  RCCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));  // Il main essenzialmente sta continuamente in attesa di qualche messaggio che arrivi e ci sia qualcosa da eseguire, poi esegue
}

void subscription_callback(const void *msgin) {
  prev_cmd_time = millis();
}

// function which controlles the motor
void MotorControll_callback(rcl_timer_t *timer, int64_t last_call_time) {
  // left_vel_estimator->estimate_velocity();
  // digitalWrite(ORANGE, !digitalRead(ORANGE));                                            //ELIMINIAMO MUNNIZZA
  float linearVelocity;
  float angularVelocity;
  // linear velocity and angular velocity send cmd_vel topic
  linearVelocity = msg.linear.x;
  angularVelocity = msg.angular.z;
  // linear and angular velocities are converted to leftwheel and rightwheel velocities
  float vR = (linearVelocity + WHEEL_BASE * angularVelocity / 2) / wheel_radius * 60 / (2 * PI);
  float vL = (linearVelocity - WHEEL_BASE * angularVelocity / 2) / wheel_radius * 60 / (2 * PI);

  float currentRpmL = leftWheel.getRpm();
  float currentRpmR = rightWheel.getRpm();
  // pid controlled is used for generating the pwm signal
  float actuating_signal_LW = leftWheel.pid(vL, currentRpmL);
  float actuating_signal_RW = rightWheel.pid(vR, currentRpmR);
  if (vL == 0 && vR == 0) {
    leftWheel.stop();
    rightWheel.stop();
    actuating_signal_LW = 0;
    actuating_signal_RW = 0;
  } else {
    rightWheel.moveBase(actuating_signal_RW, threshold);
    leftWheel.moveBase(actuating_signal_LW, threshold);
  }
  // odometry
  float average_rps_x = ((float)(currentRpmL + currentRpmR) / 2) / 60.0;  // RPS
  float linear_x = average_rps_x * wheel_circumference_;                  // m/s
  float average_rps_a = ((float)(-currentRpmL + currentRpmR) / 2) / 60.0;
  float angular_z = (average_rps_a * wheel_circumference_) / (wheels_y_distance_ / 2.0);  //  rad/s
  float linear_y = 0;
  unsigned long now = millis();
  float vel_dt = (now - prev_odom_update) / 1000.0;
  prev_odom_update = now;
  odometry.update(  // Ci servità lato ros per implementare l'odometria come da pacchetto slide 5, slide 6 (RAGA ASSURDO MI HA COMPLETATO COPILOT IL 6 MA COME FACEVA A SAPERLO AHAHAHAHA)
    vel_dt,
    linear_x,
    linear_y,
    angular_z);
  publishData();
}

// function which publishes wheel odometry.
void publishData() {
  odom_msg = odometry.getData();
  ;

  struct timespec time_stamp = getTime();

  odom_msg.header.stamp.sec = time_stamp.tv_sec;
  odom_msg.header.stamp.nanosec = time_stamp.tv_nsec;
  RCSOFTCHECK(rcl_publish(&odom_publisher, &odom_msg, NULL));
}
void syncTime()  // MA QUESTO CI SERVIRà????
{
  // get the current time from the agent
  unsigned long now = millis();
  RCCHECK(rmw_uros_sync_session(10));
  unsigned long long ros_time_ms = rmw_uros_epoch_millis();
  // now we can find the difference between ROS time and uC time
  time_offset = ros_time_ms - now;
}

struct timespec getTime() {
  struct timespec tp = { 0 };
  // add time difference between uC time and ROS time to
  // synchronize time with ROS
  unsigned long long now = millis() + time_offset;
  tp.tv_sec = now / 1000;
  tp.tv_nsec = (now % 1000) * 1000000;
  return tp;
}