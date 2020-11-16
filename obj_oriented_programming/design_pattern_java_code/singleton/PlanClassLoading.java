package design_pattern_java_code.singleton;

// Singleton instantiation at the time of classloading
public class PlanClassLoading {

    private Integer testInt;

    public static PlanClassLoading planClassLoading = new PlanClassLoading();

    PlanClassLoading(){
        if (planClassLoading != null){
            throw new ExceptionInInitializerError("Singleton Obj cannot be instantaited twice.");
        }
        this.testInt = 666;
    }

    public PlanClassLoading getPlanClassLoading(){
        return planClassLoading;
    }

    public void setTestInt(Integer newInt){
        this.testInt = newInt;
    }

    public Integer getTestInt() {
        return testInt;
    }

    public static void main(String args[]){

        System.out.println(planClassLoading.getPlanClassLoading().getTestInt());

    }
}
