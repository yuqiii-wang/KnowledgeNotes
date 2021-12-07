package oop_example_java_code.launcher;

import oop_example_java_code.text.Note;
import oop_example_java_code.text.TextNote;
import oop_example_java_code.text.TextAndImageNote;
import oop_example_java_code.service.NoteStore;
import oop_example_java_code.service.NoteStoreInterface;

/*
    Launcher by which user executes the main()
*/
public class Launcher {

    private NoteStoreInterface noteStore;

    public static void main(String args[]) {

        Launcher launch = new Launcher();

        TextNote textNote01 = new TextNote("Jason", 1, "Java is sexy and cool.");
        TextNote textNote02 = new TextNote("Yuqi", 2, "I love reading books.");
        TextAndImageNote textImgNote01 = new TextAndImageNote("Alex", 1, "That's great.", "//foo/bar/img1.png");
        TextAndImageNote textImgNote02 = new TextAndImageNote("John", 2, "It should be fun.", "//foo/bar/img2.png");

        launch.setNoteStore(new NoteStore());

        launch.getNoteStore().storeNote(textNote01);
        launch.getNoteStore().storeNote(textNote02);
        launch.getNoteStore().storeNote(textImgNote01);
        launch.getNoteStore().storeNote(textImgNote02);

        launch.displayTextNotes();
        launch.displayTextAndImageNote();
    }

    public void displayTextNotes() {
        this.getNoteStore().getAllTextNotes();
    }

    public void displayTextAndImageNote() {
        this.getNoteStore().getAllTextAndImageNotes();
    }

    public NoteStoreInterface getNoteStore() {
        return noteStore;
    }

    public void setNoteStore(NoteStoreInterface noteStore) {
        this.noteStore = noteStore;
    }
}
