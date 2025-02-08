package design_pattern_java_code.chain_of_responsibility;

public class FinalLevelProcess extends Process {

    public FinalLevelProcess(int level){
        super.setLevel(level);
    }

    @Override
    public void illustrateProcess() {
       System.out.println("This is the final process.");
    }
}
