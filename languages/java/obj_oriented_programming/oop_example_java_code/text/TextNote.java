package oop_example_java_code.text;

/*
    Text notes
*/
public class TextNote {
    private String name;
    private int id;
    private String note;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getNote() {
        return note;
    }

    public void setNote(String note) {
        this.note = note;
    }

    @Override
    public String toString() {
        return "TextNote [id=" + id + ", note=" + note + "]";
    }

    public TextNote(String name, int id, String note) {
        this.name = name;
        this.id = id;
        this.note = note;
    }
}
