package design_pattern_java_code.chain_of_responsibility;

public abstract class Process {
    private int firstLevelProcess = 1;  
    private int secondLevelProcess = 2;
    private int finalLevelProcess = 3;

    private int level = -1;

    private Process nextLevelProcess;

    public abstract void illustrateProcess();

    public void showProcess(int level){
        if (this.level <= level){
            illustrateProcess();
        }
        if (nextLevelProcess != null){
            nextLevelProcess.showProcess(level);
        }
    }

    public int getFirstLevelProcess() {
        return firstLevelProcess;
    }

    public void setFirstLevelProcess(int firstLevelProcess) {
        this.firstLevelProcess = firstLevelProcess;
    }

    public int getSecondLevelProcess() {
        return secondLevelProcess;
    }

    public void setSecondLevelProcess(int secondLevelProcess) {
        this.secondLevelProcess = secondLevelProcess;
    }

    public int getFinalLevelProcess() {
        return finalLevelProcess;
    }

    public void setFinalLevelProcess(int finalLevelProcess) {
        this.finalLevelProcess = finalLevelProcess;
    }

    public int getLevel() {
        return level;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public Process getNextLevelProcess() {
        return nextLevelProcess;
    }

    public void setNextLevelProcess(Process nextLevelProcess) {
        this.nextLevelProcess = nextLevelProcess;
    }

}
