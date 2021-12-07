package design_pattern_java_code.prototype;

import java.util.GregorianCalendar;

public class PlanMain {
    public static void main(String args[]){
        
        // init some vals
        String planName = "HK Holiday Plan";
        GregorianCalendar startDate = new GregorianCalendar(2019, 10, 1);
        GregorianCalendar deadline = new GregorianCalendar(2019, 10, 7);

        Plan hkHoliday = new Plan(planName, startDate, deadline);
        Plan hkHoliday2 = (Plan)hkHoliday.getClone();

        System.out.println(hkHoliday2);
    }
}
