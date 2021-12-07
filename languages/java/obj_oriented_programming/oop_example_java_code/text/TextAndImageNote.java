package oop_example_java_code.text;

/*
    Text and Image Notes
*/
public class TextAndImageNote implements Note{

    private String name;
    private int id;
    private String note;
    private String imageUrl;

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

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    @Override
    public String toString() {
        return "TextAndImageNote [id=" + id + ", note=" + note + " imageUrl=" + imageUrl + "]";
    }

    public TextAndImageNote(String name, int id, String note, String imageUrl) {
        this.name = name;
        this.id = id;
        this.note = note;
        this.imageUrl = imageUrl;
    }
    
}
