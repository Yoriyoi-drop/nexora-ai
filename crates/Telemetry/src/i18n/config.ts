import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      id: {
        translation: {
          // App title and common
          appTitle: "Nexora — Observatorium Kognisi Terdistribusi",
          loading: "Memuat...",
          
          // Dashboard
          dashboard: "Dasbor",
          overview: "Ikhtisar",
          metrics: "Metrik",
          analytics: "Analitik",
          
          // Common actions
          refresh: "Segarkan",
          settings: "Pengaturan",
          logout: "Keluar",
          save: "Simpan",
          cancel: "Batal",
          delete: "Hapus",
          edit: "Edit",
          
          // Status
          active: "Aktif",
          inactive: "Tidak Aktif",
          online: "Online",
          offline: "Offline",
          
          // Time
          today: "Hari Ini",
          yesterday: "Kemarin",
          lastWeek: "Minggu Lalu",
          lastMonth: "Bulan Lalu",
          
          // Error messages
          error: "Terjadi kesalahan",
          tryAgain: "Coba lagi",
          connectionError: "Koneksi gagal",
        }
      },
      en: {
        translation: {
          appTitle: "Nexora — Distributed Cognition Observatory",
          loading: "Loading...",
          dashboard: "Dashboard",
          overview: "Overview",
          metrics: "Metrics",
          analytics: "Analytics",
          refresh: "Refresh",
          settings: "Settings",
          logout: "Logout",
          save: "Save",
          cancel: "Cancel",
          delete: "Delete",
          edit: "Edit",
          active: "Active",
          inactive: "Inactive",
          online: "Online",
          offline: "Offline",
          today: "Today",
          yesterday: "Yesterday",
          lastWeek: "Last Week",
          lastMonth: "Last Month",
          error: "An error occurred",
          tryAgain: "Try again",
          connectionError: "Connection failed",
        }
      }
    },
    lng: 'id', // Default language: Indonesian
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
