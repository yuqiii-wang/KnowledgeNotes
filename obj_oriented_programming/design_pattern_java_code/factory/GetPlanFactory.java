package design_pattern_java_code.factory;

public class GetPlanFactory {

    private final String DOMESTIC_PLAN = "DOMESTIC_PLAN";
    private final String FOREIGN_PLAN = "FOREIGN_PLAN";
    
    public static void main(String args[]){
        GetPlanFactory factory = new GetPlanFactory();
        // Plan selectedPlan = factory.getPlan("");
        Plan selectedPlan = factory.getPlan("DOMESTIC_PLAN");

        if (selectedPlan != null){
            selectedPlan.setRate(5);
            selectedPlan.calculateBill(10);
        }
    }

    public Plan getPlan(String planType) {

        switch (planType){
            case DOMESTIC_PLAN:
                return new DomesticPlan();
            case FOREIGN_PLAN:
                return new ForeignPlan();
            default:
                return null;
        }
    }
}
