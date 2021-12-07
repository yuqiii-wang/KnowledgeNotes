package design_pattern_java_code.singleton;


// Singleton instantiation when required
public class PlanOnRuntime {

    private static PlanOnRuntime planOnRuntime = null;
    
    public static PlanOnRuntime getPlanOnRuntime(){
        if (planOnRuntime == null){
            synchronized (PlanOnRuntime.class) {
                //instance will be created at request time
                planOnRuntime = new PlanOnRuntime();  
            }       
         }
       return planOnRuntime;  
    }

    public static void main(String args[]){
        PlanOnRuntime.getPlanOnRuntime();
    }
}
