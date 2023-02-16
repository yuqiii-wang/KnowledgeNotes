#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#define NUMBER 2

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;


void *ThreadWork1(void *arg)
{
  int *p = (int*)arg;
  pthread_mutex_lock(&mutex1);
  
  sleep(2);
  
  pthread_mutex_lock(&mutex2);
  pthread_mutex_unlock(&mutex2);
  pthread_mutex_unlock(&mutex1);
  return NULL;
}

void *ThreadWork2(void *arg)
{
  int *p = (int*)arg;
  pthread_mutex_lock(&mutex2);
  
  sleep(2);
  
  pthread_mutex_lock(&mutex1);
  pthread_mutex_unlock(&mutex1);
  pthread_mutex_unlock(&mutex2);
  return NULL;
}
int main()
{
  pthread_t tid[NUMBER];
  pthread_mutex_init(&mutex1,NULL);
  pthread_mutex_init(&mutex2,NULL);
  int i = 0;
  int ret = pthread_create(&tid[0],NULL,ThreadWork1,(void*)&i);
  if(ret != 0)
  {
    perror("pthread_create");
    return -1;
  }
  ret = pthread_create(&tid[1],NULL,ThreadWork2,(void*)&i);
  if(ret != 0)
  {
    perror("pthread_create");
    return -1;
  }

  pthread_join(tid[0],NULL);
  pthread_join(tid[1],NULL);
  pthread_mutex_destroy(&mutex1);
  pthread_mutex_destroy(&mutex2);
  while(1)
  {
    printf("i am main work thread\n");
    sleep(1);
  }
  return 0;
}