#include <stdio.h>
#define MAX_STACK_SIZE 50

int stack[MAX_STACK_SIZE];     
int top = -1;            

int is_instance_stack_empty() {

   if(top == -1)
      return 1;
   else
      return 0;
}
   
int is_instance_stack_full() {

   if(top == MAX_STACK_SIZE)
      return 1;
   else
      return 0;
}

int pop_instance() {
   int data;
	
   if(!is_instance_stack_empty()) {
      data = stack[top];
      top = top - 1;   
      return data;
   } else {
      return 1;
   }
}

int push_instance() {
   if(!is_instance_stack_full()) {
      top = top + 1;   
      stack[top] = 1;
   } else {
      return 1;
   }
   return 0;
}
