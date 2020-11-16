package design_pattern_java_code.observer;

import java.util.Observable;  
import java.util.Observer;  

public class ListenerOne implements Observer{
    private Integer state;

    public void update(Observable obs, Object arg) {  
        if (arg instanceof Integer && arg != null){
            state = (Integer)arg;
            System.out.println("Listener One's State: " + state);
        }
    }
}
