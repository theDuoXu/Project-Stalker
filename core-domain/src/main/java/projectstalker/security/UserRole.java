package projectstalker.security;

public enum UserRole {
    ADMIN,
    ANALYST,
    TECHNICIAN,
    OFFICER,
    GUEST;

    public boolean isStaff() {
        return this != GUEST;
    }
}