package design_pattern_java_code.chain_of_responsibility;

public class ProcessMain {
    public static final int FIRST_LVL = 1;
    public static final int SECOND_LVL = 2;
    public static final int FINAL_LVL = 3;

    public Process chainedProcess(){
        FirstLevelProcess firstLvlProc = new FirstLevelProcess(FIRST_LVL);

        SecondLevelProcess secondLvlProc = new SecondLevelProcess(SECOND_LVL);
        firstLvlProc.setNextLevelProcess(secondLvlProc);

        FinalLevelProcess finalLvlProc = new FinalLevelProcess(FINAL_LVL);
        secondLvlProc.setNextLevelProcess(finalLvlProc);

        return firstLvlProc;
    }

    public static void main(String args[]){
        ProcessMain procMain = new ProcessMain();
        Process chainedProc = procMain.chainedProcess();

        chainedProc.showProcess(FIRST_LVL);
        System.out.println("=====================");
        chainedProc.showProcess(SECOND_LVL);
        System.out.println("=====================");
        chainedProc.showProcess(FINAL_LVL);
        System.out.println("=====================");
    }
}
