package design_pattern_java_code.prototype;

import java.util.GregorianCalendar;

public class Plan implements PlanPrototype{

    private String planName;
    private GregorianCalendar deadline;
    private GregorianCalendar startGregorianCalendar;

    Plan(String planName, GregorianCalendar startGregorianCalendar, GregorianCalendar deadline){
        this.setPlanName(planName);
        this.setStartGregorianCalendar(startGregorianCalendar);
        this.setDeadline(deadline);
    }

    @Override
    public PlanPrototype getClone() {
        return new Plan(planName, startGregorianCalendar, deadline);
    }

    public String getPlanName() {
        return planName;
    }

    public void setPlanName(String planName) {
        this.planName = planName;
    }

    public GregorianCalendar getDeadline() {
        return deadline;
    }

    public void setDeadline(GregorianCalendar deadline) {
        this.deadline = deadline;
    }

    public GregorianCalendar getStartGregorianCalendar() {
        return startGregorianCalendar;
    }

    public void setStartGregorianCalendar(GregorianCalendar startGregorianCalendar) {
        this.startGregorianCalendar = startGregorianCalendar;
    }

    @Override
    public String toString() {
        return "Plan [deadline=" + deadline + ", planName=" + planName + ", startGregorianCalendar="
                + startGregorianCalendar + "]";
    }
}
