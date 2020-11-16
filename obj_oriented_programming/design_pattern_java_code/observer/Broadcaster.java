package design_pattern_java_code.observer;

import java.io.BufferedReader;  
import java.io.IOException;  
import java.io.InputStreamReader;  
import java.util.Observable;  
  
public class Broadcaster extends Observable implements Runnable {  
    @Override
     public void run(){
        setChanged();  
        Integer intResponse = 1;
        notifyObservers(intResponse);  
    }  

    public static void main(String args[]){
        Broadcaster broadcaster = new Broadcaster();
        broadcaster.run();
    }
}