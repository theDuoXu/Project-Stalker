#include <jni.h>

/*
 * Mteodo para forzar compilación
 */
extern "C" JNIEXPORT void JNICALL
Java_projectstalker_Main_initialize(JNIEnv *env, jclass clazz) {
    // De momento, esta función puede estar vacía.
    // Su simple existencia forzará la compilación de la librería.
}